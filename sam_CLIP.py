import torch
import cv2
import numpy as np
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import argparse

class SAM_CLIP_Segmentor:
    def __init__(self, sam_checkpoint, device='cuda'):
        # 初始化SAM模型
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.96,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            output_mode="binary_mask",
        )
        
        # 初始化CLIP模型
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def segment_and_refine(self, image, object_parts=None, materials=None):
        """使用SAM分割图像，然后用CLIP进行语义优化"""
        # 1. 使用SAM生成初始掩码
        sam_masks = self.mask_generator.generate(image)
        print(f"SAM generated {len(sam_masks)} masks")
        
        # 如果没有指定对象部件和材质，直接返回SAM结果
        if not object_parts and not materials:
            return sam_masks
        
        # 2. 提取掩码并调用refine_existing_masks进行优化
        masks = [mask_data["segmentation"] for mask_data in sam_masks]
        refined_masks = self.refine_existing_masks(image, masks, object_parts, materials)
        
        # 3. 复制SAM生成的其他信息到优化后的结果中
        for i, refined_mask in enumerate(refined_masks):
            # 添加SAM生成的其他相关属性（如稳定性分数等）
            for sam_mask in sam_masks:
                if np.array_equal(refined_mask["segmentation"], sam_mask["segmentation"]):
                    for key, value in sam_mask.items():
                        if key not in refined_mask and key != "segmentation":
                            refined_mask[key] = value
                    break
        
        print(f"Refined to {len(refined_masks)} semantic masks")
        return refined_masks

    def semantic_segment(self, image):
        """
        先用SAM分割，然后用CLIP自动识别每个区域的语义内容，最后合并相似区域
        
        Args:
            image: 输入图像
            
        Returns:
            refined_masks: 优化后的掩码列表，每个掩码包含自动生成的类别信息
        """
        # 1. 使用SAM生成初始掩码
        sam_masks = self.mask_generator.generate(image)
        print(f"SAM generated {len(sam_masks)} masks")
        
        # 2. 使用CLIP为每个掩码生成描述
        mask_descriptions = {}
        
        for i, mask_data in enumerate(sam_masks):
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            
            # 提取掩码区域
            masked_image = self.extract_masked_region(image, mask)
            if masked_image is None:
                continue
            
            # 使用CLIP自动识别区域内容
            description = self.describe_image_region(masked_image)
            
            # 保存描述结果
            mask_data["category"] = description["category"]
            mask_data["material"] = description["material"]
            mask_data["confidence"] = description["confidence"]
            
            # 用于后续合并的组合键
            combined_key = f"{description['category']}_{description['material']}"
            if combined_key not in mask_descriptions:
                mask_descriptions[combined_key] = []
            mask_descriptions[combined_key].append(mask_data)
        
        # 3. 合并具有相同描述的掩码
        refined_masks = []
        
        for combined_key, masks in mask_descriptions.items():
            if len(masks) == 1:
                refined_masks.append(masks[0])
            else:
                # 按置信度排序
                sorted_masks = sorted(masks, key=lambda x: x["confidence"], reverse=True)
                
                # 合并相同描述的掩码
                combined_mask = np.zeros_like(sorted_masks[0]["segmentation"])
                for mask_data in sorted_masks:
                    combined_mask = np.logical_or(combined_mask, mask_data["segmentation"])
                
                # 创建新的掩码数据
                category, material = combined_key.split('_')
                refined_mask = {
                    "segmentation": combined_mask,
                    "area": float(combined_mask.sum()),
                    "bbox": self.compute_bbox(combined_mask),
                    "category": category,
                    "material": material,
                    "confidence": sorted_masks[0]["confidence"],
                    "original_masks": len(sorted_masks)
                }
                refined_masks.append(refined_mask)
        
        print(f"Refined to {len(refined_masks)} semantic masks")
        return refined_masks

    def describe_image_region(self, image):
        """
        使用CLIP自动描述图像区域的内容和材质
        
        Args:
            image: 掩码区域的PIL图像
            
        Returns:
            description: 包含类别和材质的描述字典
        """
        # 预定义一组常见对象部件
        common_parts = []
        
        # 预定义一组常见材质
        common_materials = [
            "wood", "metal", "plastic", "glass", "ceramic", "leather",
            "fabric", "stone", "marble", "rubber", "paper", "concrete"
        ]
        
        # 编码图像
        image_features = self.encode_image(image)
        
        # 识别部件类别
        # Instead of using predefined lists, use CLIP to generate a description
        
        # Step 1: Test different object types
        object_types = [
            "furniture", "kitchenware", "electronic device", "decoration", 
            "structural element", "plant", "container", "accessory"
        ]
        type_features = self.encode_text(object_types)
        type_similarity = self.compute_similarity(image_features, type_features)
        object_type = object_types[type_similarity.argmax().item()]
        
        # Step 2: Test if it's a whole object or a part
        whole_vs_part = ["a complete object", "a part of an object"]
        part_features = self.encode_text(whole_vs_part)
        part_similarity = self.compute_similarity(image_features, part_features)
        is_part = part_similarity.argmax().item() == 1
        
        # Step 3: Generate description based on results
        if is_part:
            # Test what specific part it might be
            parts = ["top", "bottom", "side", "front", "back", "handle", 
                "leg", "door", "lid", "surface", "edge", "corner"]
            part_prompts = [f"the {p} of a {object_type}" for p in parts]
            part_features = self.encode_text(part_prompts)
            part_similarity = self.compute_similarity(image_features, part_features)
            part_idx = part_similarity.argmax().item()
            matched_part = f"{parts[part_idx]} of {object_type}"
        else:
            # For whole objects, get more specific
            matched_part = object_type
            
        max_part_score = max(type_similarity.max().item(), part_similarity.max().item())
        
        # 识别材质
        materials_text_features = self.encode_text(common_materials)
        materials_similarity = self.compute_similarity(image_features, materials_text_features)
        max_material_idx = materials_similarity.argmax().item()
        max_material_score = materials_similarity[max_material_idx].item()
        matched_material = common_materials[max_material_idx]
        
        # 综合置信度
        confidence = (max_part_score + max_material_score) / 2
        
        return {
            "category": matched_part,
            "material": matched_material,
            "confidence": confidence
        }

    def describe_image_region_advanced(self, image):
        """使用更高级的方法描述图像区域"""
        
        # 使用预定义的复合描述符
        compound_descriptors = [
            "wooden table top", "metal table leg", "glass cup", 
            "plastic chair seat", "fabric sofa", "ceramic teapot body",
            "ceramic teapot lid", "metal teapot handle", "metal spout",
            "wooden chair back", "wooden chair leg", "metal lamp base",
            "fabric curtain", "glass window", "plastic container"
        ]
        
        # 编码图像
        image_features = self.encode_image(image)
        
        # 计算与复合描述符的相似度
        descriptors_features = self.encode_text(compound_descriptors)
        similarity = self.compute_similarity(image_features, descriptors_features)
        
        # 找到最匹配的描述
        max_idx = similarity.argmax().item()
        max_score = similarity[max_idx].item()
        matched_descriptor = compound_descriptors[max_idx]
        
        # 提取类别和材质
        materials = ["wooden", "metal", "glass", "plastic", "fabric", "ceramic"]
        words = matched_descriptor.split()
        material = words[0] if words[0] in materials else "unknown"
        
        # 移除材质词，剩下的作为类别
        category = " ".join(words[1:]) if material != "unknown" else matched_descriptor
        
        material_proper = {
            "wooden": "wood",
            "metal": "metal",
            "glass": "glass",
            "plastic": "plastic",
            "fabric": "fabric",
            "ceramic": "ceramic"
        }.get(material, material)
        
        return {
            "category": category,
            "material": material_proper,
            "confidence": max_score,
            "full_description": matched_descriptor
        }
        
    def refine_existing_masks(self, image, masks, object_parts=None, materials=None):
        """
        对已有的掩码进行语义分析和优化
        
        Args:
            image: 输入图像
            masks: 掩码列表，每个元素是一个二值掩码
            object_parts: 对象部件列表
            materials: 材质列表
        
        Returns:
            refined_masks: 优化后的掩码列表，每个掩码包含类别信息
        """
        # 准备提示文本
        prompts = []
        if object_parts:
            prompts.extend(object_parts)
        if materials:
            prompts.extend(materials)
        
        if not prompts:
            return [{"segmentation": mask, "category": f"mask_{i}"} for i, mask in enumerate(masks)]
        
        # 编码提示文本
        text_features = self.encode_text(prompts)
        
        # 分析每个掩码
        mask_groups = {}
        for i, mask in enumerate(masks):
            # 确保掩码是二值格式
            if isinstance(mask, dict) and "segmentation" in mask:
                binary_mask = mask["segmentation"]
            else:
                binary_mask = mask.astype(bool)
            
            # 转换为uint8格式
            bin_mask_uint8 = binary_mask.astype(np.uint8) * 255
            
            # 提取掩码区域
            masked_image = self.extract_masked_region(image, bin_mask_uint8)
            if masked_image is None:
                continue
                
            # 计算相似度
            image_features = self.encode_image(masked_image)
            similarity = self.compute_similarity(image_features, text_features)
            
            # 找到最匹配的类别
            max_sim_idx = similarity.argmax().item()
            max_sim_val = similarity[max_sim_idx].item()
            matched_prompt = prompts[max_sim_idx]
            
            # 创建掩码数据
            mask_data = {
                "segmentation": binary_mask,
                "area": float(binary_mask.sum()),
                "bbox": self.compute_bbox(binary_mask),
                "category": matched_prompt,
                "similarity_score": max_sim_val
            }
            
            # 分组
            if matched_prompt not in mask_groups:
                mask_groups[matched_prompt] = []
            mask_groups[matched_prompt].append(mask_data)
        
        # 合并同类掩码
        refined_masks = []
        for category, masks in mask_groups.items():
            if len(masks) == 1:
                refined_masks.append(masks[0])
            else:
                # 按相似度评分排序
                sorted_masks = sorted(masks, key=lambda x: x["similarity_score"], reverse=True)
                
                # 合并相同类别的掩码
                combined_mask = np.zeros_like(sorted_masks[0]["segmentation"])
                for mask_data in sorted_masks:
                    combined_mask = np.logical_or(combined_mask, mask_data["segmentation"])
                
                # 创建新的掩码数据
                refined_mask = {
                    "segmentation": combined_mask,
                    "area": float(combined_mask.sum()),
                    "bbox": self.compute_bbox(combined_mask),
                    "category": category,
                    "similarity_score": sorted_masks[0]["similarity_score"],
                    "original_masks": len(sorted_masks)
                }
                refined_masks.append(refined_mask)
        
        return refined_masks
    
    def analyze_masked_region(self, image, mask, prompts):
        """
        分析原图中被掩码覆盖的区域属于哪个类别
        
        Args:
            image: 原始图像
            mask: 二值掩码
            prompts: 候选类别提示列表，如 ["table top", "table leg", "chair", ...]
        
        Returns:
            matched_prompt: 最匹配的类别
            similarity_score: 相似度分数
        """
        # 确保掩码是uint8格式
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8) * 255
            
        # 提取掩码区域
        masked_image = self.extract_masked_region(image, mask)
        if masked_image is None:
            return None, 0.0
        
        # 编码文本提示
        text_features = self.encode_text(prompts)
        
        # 编码掩码图像
        image_features = self.encode_image(masked_image)
        
        # 计算相似度
        similarity = self.compute_similarity(image_features, text_features)
        
        # 找到最匹配的提示
        max_sim_idx = similarity.argmax().item()
        max_sim_val = similarity[max_sim_idx].item()
        matched_prompt = prompts[max_sim_idx]
        
        return matched_prompt, max_sim_val
    
    def extract_masked_region(self, image, mask):
        """从图像中提取掩码区域并转换为PIL图像"""
        if not np.any(mask):
            return None
            
        # 应用掩码到图像
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # 裁剪到掩码区域的边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 确保边界框有合理大小
        if rmax - rmin < 10 or cmax - cmin < 10:
            return None
            
        cropped = masked[rmin:rmax+1, cmin:cmax+1]
        
        # 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return pil_img
    
    def encode_text(self, text_prompts):
        """使用CLIP编码文本提示"""
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image):
        """使用CLIP编码图像"""
        # 预处理图像
        preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(preprocessed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def compute_similarity(self, image_features, text_features):
        """计算图像特征和文本特征之间的相似度"""
        with torch.no_grad():
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity[0]
    
    def compute_bbox(self, mask):
        """计算掩码的边界框 [x, y, width, height]"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
    
    def visualize_masks(self, image, masks, save_path=None):
        """可视化掩码结果"""
        result = image.copy()
        
        # 为不同类别生成不同颜色
        categories = set(mask["category"] for mask in masks if "category" in mask)
        colors = {}
        np.random.seed(42)  # 固定随机种子，使颜色一致
        for i, cat in enumerate(categories):
            colors[cat] = np.random.randint(0, 255, size=3).tolist()
        
        # 绘制掩码和类别
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"].astype(np.uint8)
            category = mask_data.get("category", f"mask_{i}")
            material = mask_data.get("material", "")
            color = colors.get(category, np.random.randint(0, 255, size=3).tolist())
            
            # 应用彩色掩码
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            result[mask > 0] = result[mask > 0] * 0.5 + colored_mask[mask > 0] * 0.5
            
            # 添加标签
            if "bbox" in mask_data:
                x, y, w, h = mask_data["bbox"]
                label = category
                if material:
                    label = f"{material} {label}"
                cv2.putText(result, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, result)
            
        return result

# 使用示例函数
def refine_masks_with_clip(image_path, sam_checkpoint, output_dir, object_parts=None, materials=None):
    """
    使用SAM+CLIP优化分割结果
    
    Args:
        image_path: 输入图像路径
        sam_checkpoint: SAM模型检查点路径
        output_dir: 输出目录
        object_parts: 可选，对象部件列表
        materials: 可选，材质列表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 初始化分割器
    segmentor = SAM_CLIP_Segmentor(sam_checkpoint)
    
    # 分割并优化
    refined_masks = segmentor.segment_and_refine(image, object_parts, materials)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 可视化结果
    vis_image = segmentor.visualize_masks(image, refined_masks, 
                                         save_path=os.path.join(output_dir, f"{base_name}_semantic.png"))
    
    # 保存每个分类掩码
    for i, mask_data in enumerate(refined_masks):
        category = mask_data.get("category", f"mask_{i}")
        category_safe = category.replace(" ", "_")
        mask_path = os.path.join(output_dir, f"{base_name}_{category_safe}.png")
        cv2.imwrite(mask_path, mask_data["segmentation"].astype(np.uint8) * 255)
    
    return refined_masks

def auto_segment_with_clip(image_path, sam_checkpoint, output_dir):
    """
    使用SAM分割，并让CLIP自动识别每个区域的语义
    
    Args:
        image_path: 输入图像路径
        sam_checkpoint: SAM模型检查点路径
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 初始化分割器
    segmentor = SAM_CLIP_Segmentor(sam_checkpoint)
    
    # 自动分割并识别
    refined_masks = segmentor.semantic_segment(image)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 可视化结果
    vis_image = segmentor.visualize_masks(image, refined_masks, 
                                         save_path=os.path.join(output_dir, f"{base_name}_auto_semantic.png"))
    
    # 保存每个分类掩码
    for i, mask_data in enumerate(refined_masks):
        category = mask_data.get("category", f"mask_{i}")
        material = mask_data.get("material", "unknown")
        mask_path = os.path.join(output_dir, f"{base_name}_{material}_{category}.png")
        cv2.imwrite(mask_path, mask_data["segmentation"].astype(np.uint8) * 255)
    
    return refined_masks

def analyze_existing_mask(image_path, mask_path, sam_checkpoint, prompts=None):
    """
    分析现有掩码图像的语义类别
    
    Args:
        image_path: 原始图像路径
        mask_path: 掩码图像路径
        sam_checkpoint: SAM模型检查点路径
        prompts: 候选类别提示列表，如为None则自动识别
    """
    # 加载图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128
    
    # 初始化分割器
    segmentor = SAM_CLIP_Segmentor(sam_checkpoint)
    
    if prompts:
        # 分析掩码区域
        category, score = segmentor.analyze_masked_region(image, mask, prompts)
        print(f"掩码最可能的类别是: {category}，置信度: {score:.4f}")
        return category, score
    else:
        # 自动识别
        masked_image = segmentor.extract_masked_region(image, mask.astype(np.uint8) * 255)
        if masked_image is None:
            return "unknown", 0.0
            
        description = segmentor.describe_image_region_advanced(masked_image)
        print(f"掩码区域描述: {description['material']} {description['category']}, 置信度: {description['confidence']:.4f}")
        return description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用SAM+CLIP进行图像分割和优化")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--sam_checkpoint", type=str, default="/work3/s222477/GaussianSeg/weights/sam_vit_h_4b8939.pth", help="SAM模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--object_parts", type=str, nargs="+", help="对象部件列表")
    parser.add_argument("--materials", type=str, nargs="+", help="材质列表")
    parser.add_argument("--existing_mask", type=str, help="使用现有掩码路径，而不是SAM生成掩码")
    parser.add_argument("--auto", action="store_true", help="使用CLIP自动识别区域内容，不需要预定义类别")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.existing_mask:
        # 分析现有掩码
        prompts = []
        if args.object_parts:
            prompts.extend(args.object_parts)
        if args.materials:
            prompts.extend(args.materials)
            
        analyze_existing_mask(args.image, args.existing_mask, args.sam_checkpoint, 
                            prompts if prompts else None)
    elif args.auto:
        # 使用自动模式
        auto_segment_with_clip(args.image, args.sam_checkpoint, args.output_dir)
    else:
        # 使用SAM生成并优化掩码
        refine_masks_with_clip(args.image, args.sam_checkpoint, args.output_dir, 
                              args.object_parts, args.materials)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用SAM+CLIP进行图像分割和优化")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--sam_checkpoint", type=str, default="/work3/s222477/GaussianSeg/weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--object_parts", type=str, nargs="+", help="对象部件列表")
    parser.add_argument("--materials", type=str, nargs="+", help="材质列表")
    parser.add_argument("--existing_mask", type=str, help="使用现有掩码路径，而不是SAM生成掩码")
    parser.add_argument("--auto", action="store_true", help="使用CLIP自动识别区域内容，不需要预定义类别")
    
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.existing_mask:
        # 分析现有掩码
        prompts = []
        if args.object_parts:
            prompts.extend(args.object_parts)
        if args.materials:
            prompts.extend(args.materials)
            
        analyze_existing_mask(args.image, args.existing_mask, args.sam_checkpoint, prompts)
    elif args.auto:
        # 使用自动模式
        auto_segment_with_clip(args.image, args.sam_checkpoint, args.output_dir)
    
    else:
        # 使用SAM生成并优化掩码
        refine_masks_with_clip(args.image, args.sam_checkpoint, args.output_dir, 
                              args.object_parts, args.materials)