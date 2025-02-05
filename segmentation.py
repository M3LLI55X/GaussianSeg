import torch
import torchvision.transforms as transforms
from utils.merge import Merge
# from som.seg import test_single_image
from utils.gen_config import *
import sys 
sys.path.extend(['./som', './segscripts/kaolin_scripts'])
# from som.sesam_serial import seg_serial
# import subprocess

from utils.gpt4_query import query_overall, query_refine, query_description, query_shape
# from utils.mask_postprocess import refine_masks
# from utils.texture_postprocess import region_refine, pixel_estimate
from utils.tools import *

# from scripts.kaolin_scripts.load_cfg import render_model, paint_model_w_mask

import os
import argparse

from PIL import Image
import numpy as np
import json
import cv2

def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_name", default='chair', type=str, help="Experiment name(unique)")
    parser.add_argument("--fine_grain", default=False, type=bool, help="Segmentation grain.")
    parser.add_argument("--view_num", default=1, type=int, help="Number of view points.")
    parser.add_argument("--img_dir", default='data/images', type=str, help="Directory of image file.")
    argv = parser.parse_args()
    return argv

def load_mask_image(mask_path):
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)
    return mask_array

def load_json_result(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def create_material_mapping(json_data):
    material_mapping = {}
    for key, value in json_data.items():
        material_mapping[int(key)] = value
    return material_mapping

def generate_material_array(mask_array, material_mapping):
    material_array = np.zeros_like(mask_array, dtype=object)
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            material_id = mask_array[i, j]
            material_array[i, j] = material_mapping.get(material_id, 'Unknown')
    return material_array

def load_and_resize_masks(mask_folder, target_size=(256, 256)):
    """
    读取mask文件夹中的所有图像, 并将它们调整为相同的大小。
    
    Args:
        mask_folder (str): 掩码文件所在的文件夹路径
        target_size (tuple): 目标尺寸 (宽, 高)
    
    Returns:
        merged_mask (np.array): 合并后的mask矩阵
    """
    # 初始化转换操作：调整大小和转换为张量
    resize_transform = transforms.Resize(target_size)
    
    # 初始化一个与目标大小相同的全零矩阵，作为最终合并的掩码矩阵
    merged_mask = np.zeros(target_size, dtype=np.int32)
    
    # 分割区域编号从0开始
    label_counter = 1  # 0留给背景

    # 遍历文件夹中的所有mask图像
    for mask_file in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_file)
        
        # 读取mask图像
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图
        mask_resized = resize_transform(mask)  # 调整尺寸

        # 将PIL图像转换为NumPy数组
        mask_array = np.array(mask_resized)

        # 找到当前mask中非零区域，并赋予新的label值
        mask_array = (mask_array > 0).astype(np.int32) * label_counter
        
        # 更新label_counter，每次处理完一个mask后，递增
        label_counter += 1

        # 将当前mask合并到最终的合并矩阵中
        merged_mask = np.where(mask_array > 0, mask_array, merged_mask)
    
    return merged_mask

def load_all_mask_images(mask_folder):
    mask_arrays = {}
    for root, _, files in os.walk(mask_folder):
        for file in files:
            if file.endswith('_mask.png'):
                k = int(file.split('_')[0])  # 提取k值
                mask_path = os.path.join(root, file)
                mask_image = Image.open(mask_path)
                mask_array = np.array(mask_image)
                mask_arrays[k] = mask_array
    return mask_arrays

def create_large_matrix(mask_arrays):
    # 假设所有mask图像的尺寸相同
    sample_mask = next(iter(mask_arrays.values()))
    height, width = sample_mask.shape
    large_matrix = np.zeros((height, width), dtype=int)
    return large_matrix

def merge_masks_into_large_matrix(mask_arrays, large_matrix):
    for k, mask_array in mask_arrays.items():
        large_matrix[mask_array > 0] = k  # 将mask中非零位置的值设为k
    return large_matrix

def create_color_mapping(matrix):
    unique_values = np.unique(matrix)
    color_mapping = {}
    np.random.seed(0)  # 固定随机种子以确保颜色一致
    for value in unique_values:
        color_mapping[value] = tuple(np.random.randint(0, 256, size=3))
    return color_mapping

def generate_color_image(matrix, color_mapping):
    height, width = matrix.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in color_mapping.items():
        color_image[matrix == value] = color
    return color_image

def save_color_image(color_image, output_path):
    image = Image.fromarray(color_image)
    image.save(output_path)


def process_mask_list(image, mask_list, mean_threshold, stddev_threshold):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    mask_features = []

    # calculate the color features
    for mask in mask_list:
        mean, stddev = cv2.meanStdDev(image, mask=mask)
        mask_features.append((mean, stddev))

    # initialize clustering labels
    labels = list(range(len(mask_list)))

    # merge masks with similar albedo
    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):
            if labels[i] != labels[j]:
                mean1, stddev1 = mask_features[i]
                mean2, stddev2 = mask_features[j]
                if np.allclose(mean1, mean2, atol=mean_threshold) and np.allclose(stddev1, stddev2, atol=stddev_threshold):
                    target_label = labels[i]
                    for k in range(len(labels)):
                        if labels[k] == labels[j]:
                            labels[k] = target_label
    
    merged_mask_list = [np.zeros_like(mask_list[0]) for _ in range(len(mask_list))]
    for label in set(labels):
        for i, mask_label in enumerate(labels):
            if mask_label == label:
                merged_mask_list[label] = np.bitwise_or(merged_mask_list[label], mask_list[i])

    merged_mask_list = [mask for mask in merged_mask_list if not np.all(mask == 0)]
    return merged_mask_list

def refine_masks(result_path, view, leave_index=None):
    for k in range(1):
        if leave_index is not None:
            if k != leave_index: continue
        mask_images_dir = f'{result_path}'
        ori_image_pth = f'/work3/s222477/GaussianSeg/logs/render_results/render_{view}.png'
        items = os.listdir(mask_images_dir)
        if not items: continue

        folder_path =  f'/work3/s222477/GaussianSeg/logs/clean/{view}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        files = os.listdir(folder_path)
        # clean folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # -------   Mask Refinement in 2D Image Space   ---------
        # Get masks from segmentor
        mask_list = [np.array(Image.open(os.path.join(mask_images_dir, mask_pth))).astype(bool) for mask_pth in os.listdir(mask_images_dir) ]
        masks = torch.stack([torch.from_numpy(arr) for arr in mask_list]).cuda()
        masks = masks[masks.sum((1, 2)).argsort()]
        if len(masks) == 0: continue
        image = cv2.imread(ori_image_pth)
        image = torch.tensor(image) >> 2 
        
        # Filter masks with high overlap
        for m, mask in enumerate(masks):
            union = (mask & masks[m + 1:]).sum((1, 2), True)
            masks[m + 1:] |= mask & (union > .86 * mask.sum((0, 1), True)) #判断重叠区域是否超过当前掩码90%
        
        # Identify and visualize disjoint patches
        unique, indices = masks.flatten(1).unique(return_inverse=True, dim=1)
        (cm := torch.randint(192, (unique.size(1), 3), dtype=torch.uint8))[0] = 0
        indices = indices.view_as(mask).cpu().numpy()
        unique_numbers = np.unique(indices)

        # Merge masks with similar albedo 
        mask_list = []
        for number in unique_numbers:
            if number == 0: continue
            mask = (indices == number).astype(np.uint8) * 255 
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            white_pixels = np.sum(opening == 255)
            if white_pixels < 400: continue # filter tiny region
            
            # find connected components
            n_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)

            for label in range(1, n_components):  # filter tiny region
                mask = output == label
                if mask.sum() < 300: continue
                mask_list.append((mask*255).astype('uint8'))

        mean_threshold, stddev_threshold = 5,4 # merge threshold
        mask_list_updated = process_mask_list(image, mask_list, mean_threshold, stddev_threshold)

        ori_image = Image.open(ori_image_pth).convert('RGB')
        ori_image_np = np.array(ori_image)
        # Skip white background
        mask_list_updated_new = []
        for mask in mask_list_updated:
            mask_area = mask == 255
            masked_image = ori_image_np[mask_area]
            masked_image_reshaped = masked_image.reshape(-1, 3)
            white_pixels = np.all(masked_image_reshaped == [255, 255, 255], axis=1)
            white_area_ratio = white_pixels.sum() / mask_area.sum()
            if white_area_ratio > 0.92:
                continue 
            mask_list_updated_new.append(mask)

        if len(mask_list_updated) == 0: continue 


        # -------   using SoM to mark refine masks   ---------
        if masks[0].shape[0] != ori_image_np.shape[0]:
            ori_image_np = np.array(ori_image.resize((1200,1200), Image.BICUBIC))
        
        from som.task_adapter.utils.visualizer import Visualizer
        visual = Visualizer(ori_image_np, metadata=None)

        for i, mask in enumerate(mask_list_updated_new, 1):
            img = Image.fromarray(mask)
            img.save(f'{folder_path}/{i}_mask.png')

            # demo = visual.draw_binary_mask_with_number(mask, text=str(i), label_mode='1', alpha=0.05, anno_mode=['Mask', 'Mark'])

        # im = demo.get_image() 
        # im_pil = Image.fromarray(im)
        # im_pil.save(f'{folder_path}/full.jpg')