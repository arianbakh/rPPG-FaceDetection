# Setup

1. `conda create -n fd python=3.9`
2. `conda activate fd`
3. `conda install cuda-toolkit`
4. `pip install tensorflow[and-cuda]`
5. `pip install -r requirements.txt`

# Inference

1. `python face_video.py --video-path videos/multiple_short.mp4 --output-dir outputs 2>&1 | grep -v "Skipping the delay kernel, measurement accuracy will be reduced" | grep -v "but there must be at least one NUMA node, so returning NUMA node zero"`

The `grep` part gets rid of annoying info and warning messages
