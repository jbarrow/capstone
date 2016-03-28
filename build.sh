# Download ALL of MAPS
python download.py --all

# Preprocess only the monophonic portion
python preprocess.py --dir ./data/ISOL

# Ignore if this throws an error. Just being safe.
cp ~/.theanorc ~/.theanorc-old

# Create the necessary ~/.theanorc file
echo "[global]
floatX = float32
device = gpu

[nvcc]
fastmath = True" > ~/.theanorc

# We have to use `sudo` so Theano will use the GPU.
python train.py
