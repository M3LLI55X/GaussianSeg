import sys
import os
import torch
from PIL import Image
import numpy as np
import cv2

# 添加包含 `som` 模块的目录到 Python 模块搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
sys.path.extend(['/dtu/blackhole/11/180913/seg_material/utils/mask_postprocess.py'])
from som.task_adapter.utils.visualizer import Visualizer
from mask_postprocess import refine_masks

# 定义测试函数
# def test_refine_masks(result_path, leave_index=None):
#     refine_masks(result_path, leave_index)

def resize_images(image_path,exp_name):
    # 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否成功读取
    if image is None:
        print(f"Failed to load image at {image_path}")
    else:
        # 调整图片大小
        new_size = (1200, 1200)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        # 保存放大的图片
        outsave_path = f'/dtu/blackhole/11/180913/seg_material/experiments/{exp_name}/original_image.jpg'  # 替换为你想保存图片的路径
        cv2.imwrite(outsave_path, resized_image)

# 测试函数
def Merge(img_dir,exp_name):
    # 设置结果路径
    result_path = f'/dtu/blackhole/11/180913/seg_material/experiments/{exp_name}'  # 替换为你的结果路径
    leave_index = None  # 如果只想处理特定索引的掩码，可以设置为相应的索引值
    image_path = '/dtu/blackhole/11/180913/seg_material/experiments/{exp_name}/original_image.jpg'
    # image_pth=f'{img_dir}/image_0_0.png'
    resize_images(img_dir,exp_name)
    print('Image resized')
    # 调用测试函数
    refine_masks(result_path, leave_index)

# if __name__ == "__main__":
#     main()