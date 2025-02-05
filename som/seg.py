import sys
import os

# 添加包含 `som` 模块的目录到 Python 模块搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

import torch
from PIL import Image
import numpy as np
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from som.task_adapter.semantic_sam.tasks import inference_semsam_auto

import os
import argparse
import warnings

from PIL import Image
import numpy as np
import json

# 加载模型配置和权重
semsam_cfg = "/work3/s222477/seg_material/som/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "/work3/s222477/seg_material/som/ckpts/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

# 定义测试函数
def test_single_image(image_path, exp_name):
    semsam_cfg = "/work3/s222477/seg_material/som/configs/semantic_sam_only_sa-1b_swinL.yaml"
    semsam_ckpt = "/work3/s222477/seg_material/som/ckpts/swinl_only_sam_many2many.pth"
    opt_semsam = load_opt_from_config_file(semsam_cfg)
    model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
    # image_pth=f'{image_path}/image_0_0.png'
    image = Image.open(image_path).convert('RGB')
    output_dir = f'./experiments/{exp_name}/0_masks'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 调用推理函数
    results = inference_semsam_auto(model_semsam, image, level=[3], all_classes='', all_parts='', thresh='0.0', text_size=1200, hole_scale=100, island_scale=100, semantic=True, label_mode='1', alpha=0.1, anno_mode=['Mask'], points_per_side=32, save_dir=output_dir)
    
    # 保存分割结果
    mask = results[0]['segmentation']
    mask_image = Image.fromarray(np.uint8(mask) * 255)
    # mask_image.save(os.path.join(output_dir, 'segmentation_result.png'))
    print('Segmentation result saved at:', output_dir)

# 测试单张图片
# test_image_path = '/dtu/blackhole/11/180913/seg_material/output_image.png'  # 替换为你的图片路径
# test_single_image(test_image_path)