# StableVectors

Generate Stable Vector graphics representations from text captions using diffusion probabilistic models

# Installation

```
conda create -n stablevectors python=3.7
conda activate stablevectors
conda install -y pytorch torchvision numpy scikit-image -c pytorch
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom
pip install opencv-python==4.5.4.60
```

torchvision, scikit-image

```
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```
