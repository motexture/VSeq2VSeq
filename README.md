# VSeq2VSeq
Text to video diffusion model with variable length frame conditioning for infinite length video generation.

## Installation
```
git clone https://github.com/motexture/VSeq2VSeq/
cd vseq2vseq
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage
To generate a long video from a single prompt:
```
python inference.py \
    --prompt "A stormtrooper surfing on the ocean" \
    --model "motexture/VSeq2VSeq" \
    --model-2d "stabilityai/stable-diffusion-xl-base-1.0" \
    --guidance-scale 20 \
    --image-guidance-scale 12 \
    --fps 16 \
    --sdp \
    --num-frames 24 \
    --width 384 \
    --height 192 \
    --image-width 1152 \
    --image-height 640 \
    --num-steps 30 \
    --times 8 \
    --min-conditioning-n-sample-frames 2 \
    --max-conditioning-n-sample-frames 2 \
    --device cuda \
    --save-init \
    --include-model
```
Increase the --times parameter to create even longer videos.<br>

Alternatively, you can read multiple prompts from a text file, where each line corresponds to a prompt that will be linearly interpolated with the next prompt every --prompts-interval steps during the --times loop:
```
python inference.py \
    --read-prompts-from-file \
    --prompts-file "prompts.txt" \
    --prompts-interval 4 \
    --model "motexture/VSeq2VSeq" \
    --model-2d "stabilityai/stable-diffusion-xl-base-1.0" \
    --guidance-scale 20 \
    --image-guidance-scale 12 \
    --fps 16 \
    --sdp \
    --num-frames 24 \
    --width 384 \
    --height 192 \
    --image-width 1152 \
    --image-height 640 \
    --num-steps 30 \
    --times 16 \
    --min-conditioning-n-sample-frames 2 \
    --max-conditioning-n-sample-frames 2 \
    --device cuda \
    --save-init \
    --include-model
```
Prompts file example:
```
Near the fireplace Aragorn is smoking his pipe
Near the fireplace Aragorn is looking at his blade
Aragorn starts to walk outside of his home
Aragorn encounters an orc and starts to fight with it
```

## Additional info
For best results --num-frames should be 16, 24 or 32.<br>
Higher values will result in slower motion.<br>

Base model: [damo-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)

## Credits
- [bfasenfest](https://github.com/bfasenfest) for his contribution to the creation of the model
- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for the original training and inference code
