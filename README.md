# vseq2vseq
Text to video diffusion model with variable length frame conditioning for infinite length video generation.

## Installation
```
git clone https://github.com/motexture/vseq2vseq/
cd vseq2vseq
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage
```
python inference.py \
    --prompt "A stormtrooper surfing on the ocean" \
    --model "motexture/vseq2vseq" \
    --model-2d "stabilityai/stable-diffusion-xl-base-1.0" \
    --guidance-scale 20 \
    --image-guidance-scale 12 \
    --fps 16 \
    --sdp \
    --num-frames 32 \
    --width 384 \
    --height 192 \
    --image-width 1152 \
    --image-height 640 \
    --num-steps 30 \
    --times 4 \
    --min-conditioning-n-sample-frames 4 \
    --max-conditioning-n-sample-frames 4 \
    --device cuda \
    --save-init \
    --include-model
```

Increase --times parameter to create longer videos
