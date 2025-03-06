import base64
import requests
import json
import os
from tqdm import tqdm
import time
from utils.tools import pp_dic_result, check_mat_valid
import cv2

from openai import OpenAI
import numpy as np

### 0. Query overall shape (optional)
### 1. Query maintype
### 2. Query subtype
### 3. Query best-matched description 

add_prompt = """This is a(or a group of) 3D rendered image(s) of an object using a pure texture map without PBR material information. Please use its visual characteristics (including color, pattern, object type) along with your existing knowledge to infer the materials of different parts. """
valid_mat = ['Blends', 'Ceramic', 'Concrete', 'Fabric', 'Ground', 'Leather', 'Marble', 'Metal', 'Plaster', 'Plastic', 'Stone', 'Terracotta', 'Wood', 'Misc']

prompt_overall = add_prompt + """Identify the material of each part(marked with Arabic numerals), presented in the exact form of {number_id: \"material\"}. Don't output other information. (optional list of material is [Ceramic, Concrete, Fabric, Ground, Leather, Marble, Metal, Plaster, Plastic, Stone, Terracotta, Wood, Misc], The 'Misc' category is output when nothing else matches.) """
prompt_refine = """{}\n\nSelect the most similar {} material type of number {} part of the image, according to the analysis of corresponding part material(including color, pattern, roughness, age and so on...). If you find it difficult to subdivide, just output {}. Don't output other information. Only a single word representing the category from optional list needs to be output. (optional list of material is {}). """

prompt_refine_sub = """{}\n\nLook at the material carefully of number {} part of the image, here are some descriptions about {} materials, can you tell me which is the best description match the part {} in the image?\n{}
Just tell me the final result in dict format with material name and descrption. Don't output other information.
"""
prompt_shape = add_prompt + """This is an image from twelve perspectives of an object. Please tell me a description, including what kind of object it is, what parts it has, and what materials each part is made of. No more than 50 words."""
prompt_obj_append = "\nThis is a description about the object to help you understand the overall 3D object: {}"

def process_image(image_path, api_key, prompt):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')    
    # Getting the base64 string
    base64_image = encode_image(image_path)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    client = OpenAI(api_key=api_key)
    return response


def query_shape(folder_path, img_pth, api_key, type_str=None):
    """Query overall information(e.g., shape, type) of 3D object.(help to query material)"""
    cache1_pth = f"{folder_path}/gpt4_query/shape_result.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    if not os.path.exists(cache1_pth):
        json.dump({}, open(cache1_pth, 'w'), indent=4)
    data = json.load(open(cache1_pth))
    response, count, success = '', 0, False
    if img_pth not in data.keys():
        while count < 5:
            try:
                if type_str is not None: 
                    new_prompt_shape = f"{prompt_shape} (Object type: This is a {type_str})"
                else:
                    new_prompt_shape = prompt_shape
                response = process_image(img_pth, api_key, new_prompt_shape)
                if 'error' in json.loads(response.model_dump_json()).keys():
                    print("waiting for rate limit...")
                    time.sleep(5)
                    continue
                print(f"success: query {img_pth}")
                success = True
                break
            except:
                count += 1
                print(f"error: query {img_pth}")
                time.sleep(5)
                continue
        if success:        
            data[img_pth] = json.loads(response.model_dump_json())
            json.dump(data, open(cache1_pth, 'w'), indent=4)
    if count == 5: return None
    if data[img_pth] != "":
        shape_info = data[img_pth]["choices"][0]["message"]["content"]
        print(shape_info)
        return shape_info
    else:
        return None

def query_overall(folder_path, leave_list, api_key, obj_info=None, force=False):
    """Query main material type of different parts of the object."""
    cache1_pth = f"{folder_path}/gpt4_query/cache_result_1.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True) 
    # 检查缓存文件是否存在，如果不存在或force为True则重新创建
    if not os.path.exists(cache1_pth) or force:
        json.dump({}, open(cache1_pth, 'w'), indent=4)
    data = json.load(open(cache1_pth))
    
    # 遍历所有视角文件夹
    for elev, azimuth in leave_list:
        view_dir = f"{folder_path}/clean/{elev}_{azimuth}"
        if not os.path.exists(view_dir):
            print(f"视角文件夹不存在: {view_dir}")
            continue
        # Check for full.jpg in view directory
        img_path = os.path.join(view_dir, "full.jpg")
        if not os.path.exists(img_path):
            print(f"Full image not found: {img_path}")
            continue

    
        # Read and save the image for querying
        full_img = cv2.imread(img_path)
        cv2.imwrite(img_path, full_img)
        
        # Query the material information
        loop, count = True, 0
        while loop and count < 5:
            count += 1
            try:
                new_prompt = prompt_overall
                if obj_info is not None:
                    new_prompt = new_prompt + prompt_obj_append.format(obj_info)
                response = process_image(img_path, api_key, new_prompt)
                response_json = json.loads(response.model_dump_json())
                desc = pp_dic_result(response_json["choices"][0]["message"]["content"])
                
                if desc is not None:
                    loop = False
                    for k, mat in desc.items():
                        c_mat = check_mat_valid(mat, count)
                        if c_mat is not None:
                            desc[k] = c_mat  # Valid main type
                        else:
                            loop = True  # Continue loop, ask GPT-4 again
                            break
                    if not loop:
                        print(f"Successfully queried stage1: {img_path}\n{desc}")
            except Exception as e:
                print(f"Query error stage1: {img_path}, error: {str(e)}")
                time.sleep(2)
                loop = True
                continue
        
        # Store the query result
        if count < 5 and desc is not None:
            response_json["choices"][0]["message"]["content"] = str(desc)
            data[img_path] = response_json
            json.dump(data, open(cache1_pth, 'w'), indent=4)


def query_refine(folder_path, leave_list, api_key, obj_info=None, force=False):
    """Query sub material type of different parts of the object."""
    result = json.load(open(f"{folder_path}/gpt4_query/cache_result_1.json",'r')) # maintype results
    sub_lst = json.load(open("/work3/s222477/GaussianSeg/weights/category_tree.json")) # category tree
    cache2_pth = f"{folder_path}/gpt4_query/cache_result_2.json" # subtype results (to be queried
    if not os.path.exists(cache2_pth):
        json.dump({}, open(cache2_pth, 'w'), indent=4)
    data = json.load(open(cache2_pth))

    for img_name, ann in result.items():
        leave_number = img_name.split('/')[-2]
        # breakpoint()
        # if leave_number not in leave_list:
            # continue
        
        # breakpoint()

        part_dic = ann["choices"][0]["message"]["content"]
        part_dic = pp_dic_result(part_dic)
        if part_dic is None: continue
        i, count, part_num = 0, 0, len(part_dic)

        # breakpoint()
        while i < part_num:
            count += 1
            if count > part_num + 4: break
            suffix = list(part_dic.keys())[i]
            main_type = part_dic[suffix]
            new_img_name = f"{leave_number}_part_{suffix}"
            if f"{leave_number}_{suffix}" in data.keys():
                i = i + 1
                continue
            if main_type not in sub_lst:
                main_type = 'Misc'
            sub_type = sub_lst[main_type] # subtypes to be select from
            new_prompt_refine = prompt_refine.format(add_prompt, main_type, suffix, main_type, sub_type)
            if obj_info is not None:
                new_prompt_refine = new_prompt_refine + prompt_obj_append.format(obj_info)
            # gpt4 ask sub-type
            try:
                response = process_image(img_name, api_key, new_prompt_refine)
                response_json = json.loads(response.model_dump_json())
                print(f"success stege2: query {img_name}_{suffix}")
            except:
                print(f"error stege2: query {img_name}_{suffix}")
                time.sleep(2)
                # continue
            # breakpoint()
            # check the value
            try:
                desc = response_json["choices"][0]["message"]["content"]
                if desc in sub_lst[main_type]: i = i + 1
                elif desc == 'Stone':
                    response_json["choices"][0]["message"]["content"] = 'PavingStones'
                    i = i + 1
                else: continue
            except: continue
            data[new_img_name] = response_json
            json.dump(data, open(cache2_pth, 'w'), indent=4)
            # breakpoint()


def query_description(folder_path, leave_list, api_key, obj_info=None, force=False):
    """Query matched material description of different parts of the object."""
    result_1 = json.load(open(f"{folder_path}/gpt4_query/cache_result_1.json",'r')) # maintype info
    result_2 = json.load(open(f"{folder_path}/gpt4_query/cache_result_2.json",'r')) # subtype info
    sub_des = json.load(open("/work3/s222477/GaussianSeg/weights/gpt_descriptions.json")) # highly-detailed annotation

    cache3_pth = f"{folder_path}/gpt4_query/cache_result_3.json" # matched description(to be queried)

    if not os.path.exists(cache3_pth):
        json.dump({}, open(cache3_pth, 'w'), indent=4)
    data = json.load(open(cache3_pth))

    # breakpoint()
    for img_name, ann in tqdm(result_1.items()):
        leave_number = img_name.split('/')[-2]
        # if leave_number not in leave_list:
            # continue
        part_dic = ann["choices"][0]["message"]["content"]
        part_dic = pp_dic_result(part_dic)
        if part_dic is None: continue
            
        part_num, count, i = len(part_dic), 0, 0

        # breakpoint()

        while i < part_num:
            suffix = list(part_dic.keys())[i]
            count += 1
            if count > part_num +4: break
            main_type = part_dic[suffix]
            new_img_name = leave_number+f'_part_{suffix}'
            if new_img_name in data.keys():
                i += 1
                # continue
            # breakpoint()
            # gpt4 ask specific description
            try:
                if type(result_2[new_img_name]) == dict: 
                    sub_type = result_2[new_img_name]["choices"][0]["message"]["content"]
                else: # read cache directly
                    sub_type = result_2[new_img_name]
                description = sub_des[main_type+'_'+sub_type] 
                new_prompt = prompt_refine_sub.format(add_prompt, suffix, main_type, suffix, description)
                if obj_info is not None:
                    new_prompt = new_prompt + prompt_obj_append.format(obj_info)
                response = process_image(img_name, api_key, new_prompt)
                print(f"success stege3: query {img_name}_{suffix}")
            except:
                print(f"error stege3: query {img_name}_{suffix}")
                time.sleep(2)
                # continue

            # check the value
            try:
                desc = pp_dic_result(json.loads(response.json())["choices"][0]["message"]["content"])
                if desc is not None: i+=1
                # else: continue
            except: continue
            data[new_img_name] = json.loads(response.json())
            json.dump(data, open(cache3_pth, 'w'), indent=4)
            # breakpoint()
        # breakpoint()    


def generate_material_matrices(folder_path, leave_list):
        """
        Generate material matrices for each view angle.
        Each matrix has the same dimensions as the original render,
        with each element representing the corresponding sub-material.
        """
        print("[INFO] Generating material matrices...")
        # Load material information from JSON files
        result_1 = json.load(open(f"{folder_path}/gpt4_query/cache_result_1.json", 'r'))  # Main types
        result_2 = json.load(open(f"{folder_path}/gpt4_query/cache_result_2.json", 'r'))  # Sub types
        
        # Dictionary to store material matrices for each view
        material_matrices = {}
        
        # Process each view angle
        for elev, azimuth in tqdm(leave_list):
            render_path = f"{folder_path}/render_results/render_{elev}_{azimuth}.png"
            view_dir = f"{folder_path}/clean/{elev}_{azimuth}"
            
            
            # Skip if render or directory doesn't exist
            if not os.path.exists(render_path) or not os.path.exists(view_dir):
                print(f"Skipping {elev}_{azimuth}: Missing files")
                continue
            
            # Get render image dimensions
            render_img = cv2.imread(render_path)
            height, width = render_img.shape[:2]
            
            # Initialize material matrix with zeros (0 will represent background/no material)
            material_matrix = np.zeros((height, width), dtype=np.int32)
            
            # Dictionary to map part IDs to material types
            material_map = {}
            
            # Get all mask files for this view
            mask_files = sorted([f for f in os.listdir(view_dir) if f.endswith('_mask.png')])
            # breakpoint()
            # Process each part
            for mask_file in mask_files:
                part_id = mask_file.split('_')[0]  # Get part ID
                mask_path = os.path.join(view_dir, mask_file)
                
                # Find material info for this part
                query_img_path = f"{folder_path}/clean/{elev}_{azimuth}/full.jpg"
                
                # Skip if no material info found
                if query_img_path not in result_1:
                    continue
                    
                # Get the main material type
                main_type_content = result_1[query_img_path]["choices"][0]["message"]["content"]
                main_type_dict = pp_dic_result(main_type_content)
                # breakpoint()
                if main_type_dict is None:
                    continue
                # breakpoint()

                part_id_str = part_id
                part_id_int = int(part_id)

                if part_id_int in main_type_dict:
                    main_type = main_type_dict[part_id_int]
                elif part_id_str in main_type_dict:
                    main_type = main_type_dict[part_id_str]
                else:
                    print(f"警告: 部件 {part_id} 在材质字典中不存在: {main_type_dict}")
                    continue

                
                # Get the sub material type
                sub_type_key = f"{elev}_{azimuth}_part_{part_id}"
                if sub_type_key not in result_2:
                    print("didn't find sub type")
                    continue
                # breakpoint()
                if isinstance(result_2[sub_type_key], dict):
                    sub_type = result_2[sub_type_key]["choices"][0]["message"]["content"]
                else:
                    sub_type = result_2[sub_type_key]
                    
                # Store the material info in the map
                material_map[part_id] = {
                    'main_type': main_type,
                    'sub_type': sub_type
                }
                
                # Load the mask and apply to the material matrix
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height))
                
                # Assign a unique ID for this material (using the part_id as an integer)
                try:
                    material_id = int(part_id)
                except ValueError:
                    material_id = hash(part_id) % 1000  # Limit to reasonable range
                    
                # Apply mask to matrix
                material_matrix[mask > 127] = material_id
            # breakpoint()
            # Store the resulting matrix and material map
            material_matrices[(elev, azimuth)] = {
                'matrix': material_matrix,
                'material_map': material_map
            }
        
        # Create directory for material matrices
        material_dir = f"{folder_path}/material_matrices"
        os.makedirs(material_dir, exist_ok=True)
        
        # Save the material maps
        material_map_file = f"{material_dir}/material_maps.json"
        material_maps = {
            f"{elev}_{azimuth}": data['material_map']
            for (elev, azimuth), data in material_matrices.items()
        }
        
        with open(material_map_file, 'w') as f:
            json.dump(material_maps, f, indent=4)
        
        # Save the matrices as numpy files and visualizations
        for (elev, azimuth), data in material_matrices.items():
            # Create a color-coded visualization
            matrix = data['matrix']
            vis_matrix = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Assign a different color to each material ID
            # Load material color mapping
            material_colors = json.load(open("/work3/s222477/GaussianSeg/weights/material_colors.json", 'r'))
            
            unique_ids = np.unique(matrix)
            for mat_id in unique_ids:
                if mat_id == 0:  # Skip background
                    continue
                
                # Get the material name for this ID
                part_id = str(mat_id)
                if part_id in data['material_map']:
                    material = data['material_map'][part_id]['sub_type']
                    
                    # Check if material exists in color mapping
                    if material in material_colors:
                        # Use the material-specific color
                        r, g, b = material_colors[material]
                        color = [b, g, r]  # OpenCV uses BGR format
                    else:
                        # Fallback color if material not found
                        color = [0, 255, 255]  # Yellow in BGR
                else:
                    # Fallback color if part_id not in material_map
                    color = [255, 0, 0]  # Blue in BGR
                
                # Apply color to the visualization
                vis_matrix[matrix == mat_id] = color
                
            # Save visualization
            vis_file = f"{material_dir}/material_vis_{elev}_{azimuth}.png"
            cv2.imwrite(vis_file, vis_matrix)
        
        print(f"Material matrices saved to {material_dir}")
        print(f"Material maps saved to {material_map_file}")
        
        return material_matrices


if __name__ == "__main__":
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
    api_key ='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'
    folder_path='/work3/s222477/GaussianSeg/logs'
    leave_list= views
    mv_img_pth='/work3/s222477/GaussianSeg/logs/render_results/render_0_0.png'
    # GPT-4V query about material of each part
    obj_info = query_shape(folder_path, mv_img_pth, api_key)
    query_overall(folder_path, leave_list, api_key, obj_info)
    query_refine(folder_path, leave_list, api_key, obj_info)
    query_description(folder_path, leave_list, api_key, obj_info, force=True)
    material_matrices = generate_material_matrices(folder_path, leave_list)
