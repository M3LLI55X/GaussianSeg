# GaussianSeg
## pipeline

记录'https://docs.google.com/document/d/1C5G-zz1GJxktA-yqDoAbEAJJfyRh2vVJcT0OEjUmKcg/edit?usp=sharing'
Flow
'https://www.figma.com/design/hBCBb2nz4zmrlWOtARmQIl/Untitled?node-id=0-1&t=5xLodQL2wHXQ1cBh-1'


## Install
```
cuda/12.4 gcc/11.5.0-binutils-2.43
```

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# # To use MVdream, also install:
# pip install git+https://github.com/bytedance/MVDream

# # To use ImageDream, also install:
# pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Usage

Image-to-3D:
```
CUDA_VISIBLE_DEVICES=3 python -m ipdb main.py --config configs/image.yaml input=data/teapot_rgba.png save_path=exp --seg /work3/s222477/GaussianSeg/data/teapot_seg.png
```
