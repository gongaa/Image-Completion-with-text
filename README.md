# Image Completion with text
- our task is to inpaint missing area in an image, given an optional textual description (multiple (s,p,o) triplet).
- Dataset: Cleaned version of Visual Genome used in VTransE, the part that has overlap with COCO2017 Segmentation mask.

## How to setup conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
chmod +x Anaconda3-2018.12-Linux-x86_64.sh
./Anaconda3-2018.12-Linux-x86_64.sh
source .bashrc
conda create -n fyp -y
source activate fyp
conda install -y jupyter ipython
conda install -y pytorch torchvision -c pytorch
```

## How to run jupyter server remotely
```bash
remote$ tmux a # or tmux if there is no session
remote$ jupyter notebook --port 8008
local$ tmux a # or tmux if there is no session
local$ ssh -N -L 9009:localhost:8008 user@host
```

Then go to browser and `127.0.0.1:9009`
