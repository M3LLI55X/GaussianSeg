import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segmentation import refine_masks

import subprocess
import sys

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)
            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # # save images for visualization
                # os.makedirs('logs/imgs', exist_ok=True)
                img_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                # if self.step == self.opt.iters:
                    # cv2.imwrite(f'logs/imgs/step_{self.step}.png', img_np[..., ::-1])

                # with open("poses.txt", "a") as f:
                    # f.write(f"{pose.tolist()}\n")

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
            
            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
   
    def load_input(self, file):
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        # # 调整图像大小，归一化
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        self.input_mask = img[..., 3:] if img.shape[-1] == 4 else np.ones((self.H, self.W, 1), dtype=np.float32)
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        self.input_img = self.input_img[..., ::-1].copy()  # BGR->RGB
        
        # 若存在同名的文本提示文件，则读取
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do the last prune 
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            
            high_res = 512 # Increased from default ref_size
            # 渲染100张final_pruned的连贯视角图片保存在logs/100中
            # render_dir = 'logs/100'
            # os.makedirs(render_dir, exist_ok=True)
            # num_views = 100
            # for i in range(num_views):
            #     # 生成均匀变化的视角，这里以水平旋转360度为例，保持海拔角不变
            #     azimuth = (360 / num_views) * i
            #     elevation = self.opt.elevation  # 可根据需求调整
            #     pose = orbit_camera(elevation, azimuth, self.opt.radius)
            #     cur_cam = MiniCam(
            #         pose,
            #         high_res,
            #         high_res,
            #         self.cam.fovy,
            #         self.cam.fovx,
            #         self.cam.near,
            #         self.cam.far,
            #     )
            #     out = self.renderer.render(cur_cam)
            #     render = out["image"].detach().cpu().numpy()
            #     if render.ndim == 3 and render.shape[0] in [1, 3, 4]:
            #         render = np.transpose(render, (1, 2, 0))
            #     if render.shape[2] > 3:
            #         render = render[..., :3]
            #     render = (np.clip(render, 0, 1) * 255).astype(np.uint8)
            #     save_path = os.path.join(render_dir, f'render_{i:03d}.png')
            #     cv2.imwrite(save_path, render)
            #     print(f"Saved coherent view image at {save_path}")
            # render from multiple views with higher resolution
            views = [
                # front & back
                (0, 0),
                (0, 180),
                # left & right 
                (0, 90),
                (0, -90),
                # top & bottom views at 45 degrees
                (45, 0),
                (45, 90),
                (45, 180),
                (45, -90),
                (-45, 0),
                (-45, 90),
                (-45, 180),
                (-45, -90),
            ]

            all_renders = []
            os.makedirs(f'/work3/s222477/GaussianSeg/logs/render_results', exist_ok=True)
            for elevation, azimuth in views:
                pose = orbit_camera(elevation, azimuth, self.opt.radius)
                cur_cam = MiniCam(pose, high_res, high_res, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                out = self.renderer.render(cur_cam)
                render = out["image"].detach().cpu().numpy()
                
                if render.ndim == 3 and render.shape[0] in [1,3,4]:
                    render = np.transpose(render, (1, 2, 0))
                if render.shape[2] > 3:
                    render = render[..., :3]

                render = render.astype(np.float32)  
                render_bgr = render[:,:,::-1]  # RGB → BGR
                render_bgr = (render_bgr * 255).astype(np.uint8)
                all_renders.append(render)
                save_path = f'/work3/s222477/GaussianSeg/logs/render_results/render_{elevation}_{azimuth}.png'
                cv2.imwrite(save_path, render_bgr)
                print(f"Saved render img at {save_path}")

            os.makedirs(f'/work3/s222477/GaussianSeg/logs/seg_results', exist_ok=True)
            result_path = f'/work3/s222477/GaussianSeg/logs/seg_results'

            HOME = os.getcwd()
            CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = "vit_h"

            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            print("[INFO] Loading SAM model...")
            sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.96,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
                output_mode="binary_mask",  # 只输出二值mask
            )
            
            sam_result = []
            print("[INFO] Generating masks for each view...")
            for i, render in enumerate(all_renders):
                print(f"Processing view {i+1}/{len(all_renders)}: elevation={views[i][0]}, azimuth={views[i][1]}")
                try:
                    # 检查渲染结果
                    if render is None or render.size == 0:
                        print(f"Invalid render for view {i}")
                        sam_result.append([])
                        continue
                        
                    # 打印渲染图像信息以便调试
                    print(f"Render shape: {render.shape}, dtype: {render.dtype}, value range: [{render.min()}, {render.max()}]")
                    
                    masks = mask_generator.generate(render)
                    if len(masks) == 0:
                        print(f"Warning: No masks generated for view {i}")
                    print(f"Generated {len(masks)} masks")
                    sam_result.append(masks)
                except Exception as e:
                    print(f"Error processing view {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    sam_result.append([])

            # save each mask for each render
            for i, sam_masks in enumerate(sam_result):
                elevation, azimuth = views[i]
                view_dir = os.path.join(result_path, f'view_{elevation}_{azimuth}')
                os.makedirs(view_dir, exist_ok=True)
                    
                for j, mask in enumerate(sam_masks):
                    if mask['segmentation'] is None or not mask['segmentation'].any():
                        print(f"Skip invalid mask {j} for view_{elevation}_{azimuth}")
                        continue
                    mask_img = mask['segmentation'].astype(np.uint8) * 255
                    save_path = os.path.join(view_dir, f'mask_{j}.png')
                    cv2.imwrite(save_path, mask_img)

            # Create combined mask visualizations separately after all masks are saved
            print("[INFO] Creating combined mask visualizations...")
            rsual_path = f'/work3/s222477/GaussianSeg/logs/seg_results_visual'
            os.makedirs(rsual_path, exist_ok=True)           
            for i, (elevation, azimuth) in enumerate(views):
                view_dir = os.path.join(result_path, f'view_{elevation}_{azimuth}')
                mask_files = sorted(glob.glob(os.path.join(view_dir, 'mask_*.png')))
                # Create a blank colored mask image (black background)
                combined_mask = np.zeros((high_res, high_res, 3), dtype=np.uint8)
                
                # Add each mask with a unique color
                for j, mask_file in enumerate(mask_files):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is None or not mask.any():
                        continue
                        
                    # Generate a random color (BGR)
                    color = np.random.randint(50, 256, size=3).tolist()
                    
                    # Add colored mask to the combined image
                    mask_region = (mask > 127)
                    combined_mask[mask_region] = color
                
                # Save the combined mask image
                combined_path = os.path.join(rsual_path, f'{elevation}_{azimuth}.png')
                cv2.imwrite(combined_path, combined_mask)
            print(f"Created combined mask")




            # breakpoint()
            # Process each view directory to get clean masks
            for elevation, azimuth in views:  # 解包元组
                view_str = f'view_{elevation}_{azimuth}'  # 构造路径字符串
                mask_dir = os.path.join(result_path, view_str)  # 使用字符串构造路径
                if os.path.exists(mask_dir):
                    print(f"Refining masks in {view_str}...")
                    ori_view = f'{elevation}_{azimuth}' 
                    refine_masks(mask_dir, ori_view)

            clean_dir = "/work3/s222477/GaussianSeg/logs/clean"
            cclean_path =  f'/work3/s222477/GaussianSeg/logs/clean_visual'
            if not os.path.exists(cclean_path):
                os.makedirs(cclean_path, exist_ok=True)
            for i, (elevation, azimuth) in enumerate(views):
                view_dir = os.path.join(clean_dir, f'{elevation}_{azimuth}')
                mask_files = sorted(glob.glob(os.path.join(view_dir, '*_mask.png')))
                # Create a blank colored mask image (black background)
                combined_mask = np.zeros((high_res, high_res, 3), dtype=np.uint8)
                
                # Add each mask with a unique color
                for j, mask_file in enumerate(mask_files):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    color = np.random.randint(50, 256, size=3).tolist()
                    
                    # Add colored mask to the combined image
                    mask_region = (mask > 127)
                    combined_mask[mask_region] = color
                
                # Save the combined mask image
                combined_path = os.path.join(cclean_path, f'{elevation}_{azimuth}.png')
                cv2.imwrite(combined_path, combined_mask)

            print(f"Created combined mask")

            # breakpoint()



            
            try:
                gpt_test_path = "/work3/s222477/GaussianSeg/gpt_test.py"
                subprocess.run([sys.executable, gpt_test_path], check=True)
                print("[INFO] gpt_test.py执行完成")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] gpt_test.py执行失败: {e}")
            
            # folder_path='/work3/s222477/GaussianSeg/logs'
            # leave_list= views
            
            # Load valid materials
            # material_matrices = generate_material_matrices(folder_path, leave_list)
            


            # breakpoint()
            # # Create material matrices for each view based on GPT-4V results
            print("[INFO] Generating material matrices...")


            # breakpoint()



            # back-project masks to 3D
            view_list = [
                (0, 0), (0, 180),
                (0, 90), (0, -90),
                (45, 0), (45, 90), (45, 180), (45, -90),
                (-45, 0), (-45, 90), (-45, 180), (-45, -90),
            ]
            
            # 一个用于将2D mask信息反向投影到3D高斯点云的过程。通过渲染不同视角下的3D模型，生成对应的2D图像，然后使用SAM（Segment Anything Model）生成mask，并将这些mask反向投影到3D点云中，为每个3D点赋予标签。
            # 假设 3D 点存储在 renderer.gaussians.points 中，形状为 (N, 3)
            points = self.renderer.gaussians.get_xyz  # tensor on device
            points_cpu = points.detach().cpu().numpy()
            num_points = points_cpu.shape[0]

            # 初始化每个点的标签为 0（未被任何 2D mask 标记）
            labels = -torch.ones(num_points, dtype=torch.int32, device=self.device)

            # 遍历每个采样视角，将对应 refined mask 反向投影到 3D 点云
            render_resolution = 512  # 与 texture extraction 时一致
            proj_matrix = torch.from_numpy(self.cam.perspective.astype('float32')).to(self.device)
            
            # breakpoint()
            
            for view_idx, (elev, azimuth) in enumerate(view_list):
                # 根据当前视角计算相机位姿
                pose = orbit_camera(elev, azimuth, self.opt.radius)
                pose_tensor = torch.from_numpy(pose.astype('float32')).to(self.device)

                # 将 3D 点转换为齐次坐标
                ones = torch.ones((points.shape[0], 1), device=self.device)
                points_homog = torch.cat([points, ones], dim=1)  # (N, 4)

                # 从世界坐标转换到相机坐标
                inv_pose = torch.inverse(pose_tensor)
                points_cam = points_homog @ inv_pose.T  # (N, 4)

                # 投影到 clip space
                points_clip = points_cam @ proj_matrix.T  # (N, 4)
                # 执行透视除法，得到 NDC 空间坐标
                ndc = points_clip[:, :3] / (points_clip[:, 3:4] + 1e-8)

                # 将 NDC 坐标映射到像素坐标
                # NDC 范围 [-1, 1] 映射到 [0, render_resolution]
                u = ((ndc[:, 0] + 1) / 2) * render_resolution
                v = ((1 - (ndc[:, 1] + 1) / 2)) * render_resolution  # 注意 y 轴翻转

                u_np = u.detach().cpu().numpy()
                v_np = v.detach().cpu().numpy()

                # 加载当前视角对应的material_vis图像
                mask_dir = f"/work3/s222477/GaussianSeg/logs/material_matrices"
                mask_file = f"{mask_dir}/material_vis_{elev}_{azimuth}.png"
                
                # Initialize a dictionary to track label votes for each point
                if not hasattr(self, 'label_votes'):
                    self.label_votes = [{} for _ in range(num_points)]
                    
                if os.path.exists(mask_file):
                    # 加载彩色mask图像
                    mask_img = cv2.imread(mask_file, cv2.IMREAD_COLOR)
                    
                    # Calculate view confidence based on camera angle
                    view_confidence = 1.0
                    
                    # 对于每个3D点，检查投影后的像素位置的材质颜色
                    for i in range(num_points):
                        ui = int(round(u_np[i]))
                        vi = int(round(v_np[i]))
                        # 检查是否在图像范围内
                        if 0 <= ui < render_resolution and 0 <= vi < render_resolution:
                            # 获取该像素的颜色
                            color = mask_img[vi, ui]
                            # 如果不是黑色背景(有材质标记)
                            if color.any() > 0:
                                # 将颜色值转换为单一标签值(保留BGR颜色信息)
                                label_value = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
                                
                                # 累加该点对应标签的票数
                                if label_value not in self.label_votes[i]:
                                    self.label_votes[i][label_value] = 0
                                # 加权投票 (可以根据视角质量进行调整)
                                self.label_votes[i][label_value] += view_confidence
                else:
                    print(f"Warning: Material visualization not found at {mask_file}")

            # 处理完所有视角后，为每个点分配最多票数的标签
            # Define misc label (you may need to adjust this based on your specific case)
            misc_label = 8421504  # Example value (RGB 128,128,128)

            for i in range(num_points):
                if self.label_votes[i]:  # 如果该点有任何标签投票
                    # 分离misc和非misc标签的投票
                    non_misc_votes = {label: votes for label, votes in self.label_votes[i].items() 
                                     if label != misc_label}
                    
                    # 优先选择非misc标签中得票最多的
                    if non_misc_votes:
                        max_label = max(non_misc_votes.items(), key=lambda x: x[1])[0]
                    else:
                        # 如果只有misc标签，则使用所有标签中得票最多的
                        max_label = max(self.label_votes[i].items(), key=lambda x: x[1])[0]
                    
                    labels[i] = max_label
            # 将标签赋值回 renderer.gaussians
            self.renderer.gaussians.labels = labels
            print("[INFO] Reverse project complete with voting-based label assignment")

        # 加载材质颜色映射
        material_colors_file = "/work3/s222477/GaussianSeg/weights/material_colors.json"
        if os.path.exists(material_colors_file):
            with open(material_colors_file, 'r') as f:
                material_colors_map = json.load(f)
        else:
            print(f"Warning: Material colors file not found at {material_colors_file}")
            material_colors_map = {}

        # 确保点云和标签已经存在
        points = self.renderer.gaussians.get_xyz.detach()
        labels_tensor = self.renderer.gaussians.labels.detach()
        if points is None or labels_tensor is None:
            print("there is no label。")
        else:
            # 创建一个颜色映射，将标签值映射回原始RGB颜色
            label_colors = {}
            for lab in labels_tensor.unique().cpu().numpy():
                if lab < 0:
                    continue
                # 从标签值中提取原始BGR颜色
                b = (lab >> 16) & 0xFF
                g = (lab >> 8) & 0xFF
                r = lab & 0xFF
                # 转换为RGB格式
                color_key = f"{r},{g},{b}"
                # 尝试从material_colors.json获取对应的材质颜色
                if color_key in material_colors_map:
                    label_colors[int(lab)] = material_colors_map[color_key]
                else:
                    # 如果在JSON中找不到，则使用提取的原始颜色
                    label_colors[int(lab)] = [r, g, b]
            
            default_color = [200, 200, 200]  # 对于未标记的点

            # 将3D点投影到2D平面，使用当前摄像机参数
            points_cpu = points.cpu().numpy()  # (N, 3)
            N = points_cpu.shape[0]
            ones = np.ones((N, 1), dtype=np.float32)
            points_homog = np.concatenate([points_cpu, ones], axis=1)  # (N, 4)

            # 使用 self.cam.pose 和 self.cam.perspective 进行投影
            pose = self.cam.pose            # (4, 4) numpy数组
            proj = self.cam.perspective.astype(np.float32)
            inv_pose = np.linalg.inv(pose)

            points_cam = points_homog @ inv_pose.T  # 转换到摄像机坐标系
            points_clip = points_cam @ proj.T       # 转换到裁剪空间
            ndc = points_clip[:, :3] / (points_clip[:, 3:4] + 1e-8)  # 归一化设备坐标

            # 将NDC坐标映射到像素坐标
            u = ((ndc[:, 0] + 1) / 2) * self.W
            v = ((1 - (ndc[:, 1] + 1) / 2)) * self.H

            # 创建一个空白画布（白色背景）
            canvas = np.full((self.H, self.W, 3), 255, dtype=np.uint8)

            # 根据标签给每个点上色，并在画布上绘制小圆点
            for i in range(N):
                ui = int(round(u[i]))
                vi = int(round(v[i]))
                if ui < 0 or ui >= self.W or vi < 0 or vi >= self.H:
                    continue
                lab = int(labels_tensor[i].item())
                if lab >= 0:  # 只处理有效的标签
                    # 直接从标签值中提取BGR颜色
                    b = (lab >> 16) & 0xFF
                    g = (lab >> 8) & 0xFF
                    r = lab & 0xFF
                    color = [b, g, r]  # OpenCV使用BGR顺序
                else:
                    color = default_color
                cv2.circle(canvas, (ui, vi), 1, color, -1)

            # 保存渲染出的点云图像
            output_path = os.path.join(self.opt.outdir, "point_cloud_labels.png")
            cv2.imwrite(output_path, canvas)
            print(f"已保存点云标记图像：{output_path}")

        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import glob
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    # parser.add_argument("--seg", required=True, help="path to the segmentation file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    # opt.seg = args.seg
    gui = GUI(opt)

    if opt.gui:
        # gui.render()
        pass
    else:
        gui.train(opt.iters)