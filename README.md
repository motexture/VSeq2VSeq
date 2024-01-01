# VSeq2VSeq
Text to video diffusion model with variable length frame conditioning for infinite length video generation.

## Installation
```
git clone https://github.com/motexture/VSeq2VSeq/
cd VSeq2VSeq
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage
To generate a long video from a single prompt:
```
python inference.py \
    --prompt "A stormtrooper surfing on the ocean" \
    --video-diffusion-model "motexture/VSeq2VSeq" \
    --image-diffusion-model "stabilityai/stable-diffusion-xl-base-1.0" \
    --guidance-scale 12 \
    --fps 16 \
    --sdp \
    --num-frames 16 \
    --num-conditioning-frames 4 \
    --width 320 \
    --height 192 \
    --image-width 1280 \
    --image-height 768 \
    --num-steps 50 \
    --times 8 \
    --device cuda \
```
Increase the --times parameter to create even longer videos.<br>

Alternatively, you can read multiple prompts from a text file, where each line corresponds to a prompt that will be linearly interpolated with the next prompt every --prompts-interval steps during the --times loop:
```
python inference.py \
    --read-prompts-from-file \
    --prompts-file "prompts.txt" \
    --prompts-interval 4 \
    --video-diffusion-model "motexture/VSeq2VSeq" \
    --image-diffusion-model "stabilityai/stable-diffusion-xl-base-1.0" \
    --guidance-scale 12 \
    --fps 16 \
    --sdp \
    --num-frames 16 \
    --num-conditioning-frames 4 \
    --width 320 \
    --height 192 \
    --image-width 1280 \
    --image-height 768 \
    --num-steps 50 \
    --times 16 \
    --device cuda \
```
Prompts file example:
```
Near the fireplace Aragorn is smoking his pipe
Near the fireplace Aragorn is looking at his blade
Aragorn starts to walk outside of his home
Aragorn encounters an orc and starts to fight with it
```

Base model: [damo-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)

## Credits
- [bfasenfest](https://github.com/bfasenfest) for his contribution to the creation of the model
- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for the original training and inference code
