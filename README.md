# Setup

1. `conda create -n fd python=3.9`
2. `conda activate fd`
3. `conda install cuda-toolkit`
4. `pip install tensorflow[and-cuda]`
5. `pip install -r requirements.txt`

Tested on the following environment:

1. Single Nvidia GeForce RTX 3090 24GB GPU; Ubuntu 24.04; Nvidia driver 555.52.04; CUDA 12.5 (outside conda env); conda 23.1.0; Python 3.9.19 (inside conda env).

# Inference

1. `python face_video.py --video-path videos/multiple_short.mp4 --output-dir outputs --num-processes 16`

Adjust number of processes based on your GPU memory, otherwise processes fail.
