# Setup

1. `conda create -n fd python=3.9`
2. `conda activate fd`
3. `conda install cuda-toolkit`
4. `pip install tensorflow[and-cuda]`
5. `pip install -r requirements.txt`

# Inference

1. `TF_CPP_MIN_LOG_LEVEL=2 python face_video.py --video-path videos/vid.mp4 --output-dir outputs --save-sample`
