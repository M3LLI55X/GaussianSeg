import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from guidance.zero123_utils import Zero123

def generate_novel_views(image_path, views, output_dir="/work3/s222477/GaussianSeg/logs/zero123_views", device="cuda"):
    """
    Generate novel views of an image using Zero123 following main.py's approach.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device(device)
    
    # Load the Zero123 model using the same approach as in main.py
    print(f"[INFO] loading zero123...")
    # Use the same model key as in main.py
    guidance_zero123 = Zero123(device, model_key='ashawkey/zero123-xl-diffusers')
    print(f"[INFO] loaded zero123!")
    
    # Load and preprocess the input image
    input_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if input_img.shape[-1] == 4:  # Handle alpha channel
        input_mask = input_img[..., 3:] / 255.0
        input_img = input_img[..., :3] * input_mask + (1 - input_mask) * 255
        input_img = input_img[..., ::-1].copy()  # BGR->RGB
    else:
        input_img = input_img[..., ::-1].copy()  # BGR->RGB
    
    # Resize and convert to torch tensor
    h, w = input_img.shape[:2]
    input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    
    # Get image embeddings (as done in main.py's prepare_train)
    with torch.no_grad():
        guidance_zero123.get_img_embeds(input_img_torch)
    
    # Generate novel views
    for i, (elevation, azimuth) in enumerate(views):
        print(f"Generating view {i+1}/{len(views)}: elevation={elevation}, azimuth={azimuth}")
        
        # Generate image for the specific view
        with torch.no_grad():
            # 使用正确的方法 refine，并传递参数为列表
            output = guidance_zero123.refine(
                input_img_torch,           # 需要输入图像的张量
                elevation=[elevation],     # 注意参数需要是列表
                azimuth=[azimuth],         # 注意参数需要是列表
                radius=[2],                # 注意参数需要是列表
                strength=0,                # 从随机噪声开始生成
                steps=50,                   # 可选参数：迭代步数，影响质量
                # guidance_scale=1.0,
            )
        
        # Convert output tensor to image
        output_img = output.detach().cpu().numpy()
        # 转换输出格式，注意输出可能是 [B,C,H,W] 格式
        if output_img.ndim == 4:
            output_img = output_img[0]  # 取第一个批次
        if output_img.shape[0] == 3:  # 如果是 CHW 格式
            output_img = np.transpose(output_img, (1, 2, 0))
        output_img = (output_img * 255).astype(np.uint8)
        
        # Save the output
        output_path = os.path.join(output_dir, f"render_{elevation}_{azimuth}.png")
        cv2.imwrite(output_path, output_img[..., ::-1])  # Convert RGB to BGR for cv2
        print(f"Saved to {output_path}")
    
    print("All views generated successfully!")

if __name__ == "__main__":
    # Define the views as elevation-azimuth pairs (same as in your code)
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
    
    # Path to the input image
    image_path = "/work3/s222477/GaussianSeg/data/axe_rgba.png"
    
    # Generate the novel views
    generate_novel_views(image_path, views)