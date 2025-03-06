import json

def next_color(color):
    # 按蓝色通道递增，超出后绿通道递增，再超出后红通道递增，循环从1开始（0保留给背景）
    r, g, b = color
    b += 30
    if b > 255:
        b = 1
        g += 30
    if g > 255:
        g = 1
        r += 30
    return [r, g, b]

def generate_material_color_mapping(input_json, output_json):
    # 读取类别树
    with open(input_json, 'r') as f:
        category_tree = json.load(f)
    
    color_mapping = {}
    current_color = [1, 1, 1]
    
    # 遍历所有子材质，保证顺序一致（例如按字母排序）
    all_materials = []
    for sub_materials in category_tree.values():
        all_materials.extend(sub_materials)
    # 去重并排序
    unique_materials = sorted(set(all_materials))
    
    for material in unique_materials:
        color_mapping[material] = current_color
        current_color = next_color(current_color)
            
    # 将颜色映射保存为 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(color_mapping, f, indent=4)
    print(f"Material colors saved to {output_json}")

if __name__ == '__main__':
    input_json_path = '/work3/s222477/GaussianSeg/weights/category_tree.json'
    output_json_path = '/work3/s222477/GaussianSeg/weights/material_colors.json'
    generate_material_color_mapping(input_json_path, output_json_path)