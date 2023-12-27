# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/inference.py

import argparse
import warnings
import torch
import random
import subprocess
import json
import numpy as np
import os
import re
import PIL
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from train import export_to_video, load_primary_models, handle_memory_attention
from diffusers import TextToVideoSDPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Union
from einops import rearrange
from torch import Tensor
from tqdm import trange
from uuid import uuid4
from diffusers.utils import PIL_INTERPOLATION
from einops import rearrange

def match_histograms(source, reference):
    source = source.astype(np.uint8)

    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    src_cdf = np.cumsum(src_hist)
    src_cdf_normalized = src_cdf / src_cdf.max()

    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    ref_cdf = np.cumsum(ref_hist)
    ref_cdf_normalized = ref_cdf / ref_cdf.max()

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.abs(src_cdf_normalized[i] - ref_cdf_normalized).argmin()
        lookup_table[i] = idx

    matched_frame = lookup_table[source]
    return matched_frame

def enhance_contrast_clahe_4d(tensor, clip_limit=1.2, tile_grid_size=(1,1), gamma=0.98):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    output = np.empty_like(tensor)

    first_frame_matched = []

    for c in range(tensor.shape[3]):
        first_frame = tensor[0, :, :, c]
        first_frame_enhanced = clahe.apply(first_frame)
        first_frame_enhanced = np.power(first_frame_enhanced, gamma)
        first_frame_enhanced = np.clip(first_frame_enhanced, 0, 255)
        first_frame_matched.append(first_frame_enhanced)

    for f in range(tensor.shape[0]):
        for c in range(tensor.shape[3]):
            frame = tensor[f, :, :, c]
            enhanced_frame = clahe.apply(frame)
            enhanced_frame = np.power(enhanced_frame, gamma)
            enhanced_frame = np.clip(enhanced_frame, 0, 255)

            matched_frame = match_histograms(enhanced_frame, first_frame_matched[c])

            output[f, :, :, c] = matched_frame

    return output

def save_image(tensor, filename):
    tensor = tensor.cpu().numpy()
    tensor = tensor.transpose((1, 2, 0))
    tensor = (tensor * 255).astype('uint8')

    img = Image.fromarray(tensor)
    img.save(filename)

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def image_to_tensor(image_path):
    # Open image file
    image = Image.open(image_path).convert('RGB')

    return preprocess(image)

def interpolate_prompts(embeds1, embeds2, alpha):
    return (1 - alpha) * embeds1 + alpha * embeds2

def read_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]
    return prompts

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_video(input_file, output_file, height):
    command = ['ffmpeg',
               '-i', input_file,
               '-c:v', 'libx264',
               '-crf', '23',
               '-preset', 'fast',
               '-c:a', 'aac',
               '-b:a', '128k',
               '-movflags', '+faststart',
               '-vf', f'scale=-1:{height}',
               '-y',
               output_file]
    
    subprocess.run(command, check=True)

def get_video_height(input_file):
    command = ['ffprobe', 
               '-v', 'quiet', 
               '-print_format', 'json', 
               '-show_streams', 
               input_file]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    for stream in video_info.get('streams', []):
        if stream['codec_type'] == 'video':
            return stream['height']

    return None

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    vae.enable_slicing()

    handle_memory_attention(xformers, sdp, unet)

    return pipe

def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents

def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents

def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()

@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    num_inference_steps: int,
    guidance_scale: float,
    encode_to_latent: bool,
    num_frames: int,
    prompt_interpolation_alpha: Optional[float] = None,
    previous_prompt_embeds: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    conditioning_hidden_states: torch.Tensor = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    eta: float = 0.0
):
    device = pipe.device
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    if previous_prompt_embeds is not None:
        prompt_embeds = interpolate_prompts(previous_prompt_embeds, prompt_embeds, prompt_interpolation_alpha)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    conditioning_hidden_states = encode(pipe, conditioning_hidden_states, 1) if encode_to_latent else conditioning_hidden_states
    conditioning_hidden_states = conditioning_hidden_states.to(device)

    shape = (1, 4, num_frames, conditioning_hidden_states.shape[3], conditioning_hidden_states.shape[4])

    noisy_latents = torch.randn(shape, dtype=torch.half)
    noisy_latents = noisy_latents.to(device)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            noisy_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            conditioned_model_input = torch.cat([conditioning_hidden_states] * 2) if do_classifier_free_guidance else conditioning_hidden_states

            noisy_model_input = pipe.scheduler.scale_model_input(noisy_model_input, t)
            conditioned_model_input = pipe.scheduler.scale_model_input(conditioned_model_input, t)

            noise_pred = pipe.unet(
                noisy_model_input,
                conditioned_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            bsz, channel, frames, width, height = noisy_latents.shape
            noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

            noisy_latents = pipe.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

            noisy_latents = noisy_latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, noisy_latents)
    
    return noisy_latents, prompt_embeds

@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    read_prompts_from_file: bool,
    prompts_file: str,
    prompts_interval: int,
    negative_prompt: Optional[str] = None,
    width: int = 320,
    height: int = 192,
    image_width: int = None,
    image_height: int = None,
    model_2d: str = None,
    num_frames: int = 16,
    vae_batch_size: int = 32,
    num_steps: int = 50,
    guidance_scale: float = 20,
    image_guidance_scale: float = 12,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    times: int = 4,
    seed: Optional[int] = None,
    conditioning_hidden_states: torch.Tensor = None,
    min_conditioning_n_sample_frames: int = 4,
    max_conditioning_n_sample_frames: int = 4,
    save_init: bool = False,
    output_dir: str = "output"
):
    if seed is not None:
        set_seed(seed)

    if read_prompts_from_file:
        prompts = read_prompts(prompts_file)
        prompts = iter(prompts)
        prompt = next(prompts)
    else:
        prompts = None

    if conditioning_hidden_states is None:
        stable_diffusion_pipe = DiffusionPipeline.from_pretrained(model_2d, torch_dtype=torch.float16).to(device)
        conditioning_hidden_states = stable_diffusion_pipe(prompt=prompt, negative_prompt=negative_prompt, width=image_width, height=image_height, guidance_scale=image_guidance_scale, output_type="pt").images[0]
        
        if (save_init):
            os.makedirs(output_dir, exist_ok=True)
            unique_id = str(uuid4())[:8]
            save_image(conditioning_hidden_states, f"{output_dir}/{prompt}-{unique_id}.png")

        conditioning_hidden_states = conditioning_hidden_states.unsqueeze(0)
        conditioning_hidden_states = F.interpolate(conditioning_hidden_states, size=(height, width), mode='bilinear', align_corners=False)
        conditioning_hidden_states = conditioning_hidden_states.unsqueeze(2)

        del stable_diffusion_pipe
        torch.cuda.empty_cache()

    with torch.autocast(device, dtype=torch.half):
        pipe = initialize_pipeline(model, device, xformers, sdp)

        generator = torch.Generator().manual_seed(seed)
        
        previous_prompt_embeds = None
        video_latents = []
        for t in range(0, times):
            if t > 0 and t % prompts_interval == 0 and prompts is not None and read_prompts_from_file:
                prompt = next(prompts)
                previous_prompt_embeds = None
            
            prompt_interpolation_alpha = (t % prompts_interval) if t > 0 else 0
            
            latents, prompt_embeds = diffuse(
                pipe=pipe,
                conditioning_hidden_states=conditioning_hidden_states,
                prompt=prompt,
                negative_prompt=negative_prompt,
                encode_to_latent=True if t == 0 else False,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                prompt_interpolation_alpha=prompt_interpolation_alpha,
                previous_prompt_embeds=previous_prompt_embeds if read_prompts_from_file else None,
                generator=generator
            )
            previous_prompt_embeds = prompt_embeds
            
            video_latents.append(latents)

            concatenated_latents = torch.cat(video_latents, dim=2)
            
            random_slice = random.randint(min_conditioning_n_sample_frames, max_conditioning_n_sample_frames)
            random_slice = random_slice if random_slice < concatenated_latents.shape[2] else 1
            
            conditioning_hidden_states = concatenated_latents[:, :, -random_slice:, :, :]  

        video_latents = torch.cat(video_latents, dim=0)

        videos = decode(pipe, video_latents, vae_batch_size)

    return torch.cat(torch.unbind(videos, dim=0), dim=1)

if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="motexture/VSeq2VSeq", help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-MP", "--model-2d", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Path to the model for image generation (if init image is not provided)")
    parser.add_argument("-p", "--prompt", type=str, default="A stormtrooper is surfing on the ocean", help="Text prompt to condition on")
    parser.add_argument("-rp", "--read-prompts-from-file", default=False, action="store_true", help="File path to prompts file")
    parser.add_argument("-pf", "--prompts-file", type=str, default=None, help="Path to text file with prompts")
    parser.add_argument("-pi", "--prompts-interval", type=int, default=4, help="Interval for switching prompts when reading from file. Prompts will be interpolated linearly on this interval.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Directory to save output video to")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-FR", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-CN", "--min-conditioning-n-sample-frames", type=int, default=4, help="Total number of frames to sample for conditioning after initial video")
    parser.add_argument("-CX", "--max-conditioning-n-sample-frames", type=int, default=4, help="Total number of frames to sample for conditioning after initial video")
    parser.add_argument("-WI", "--width", type=int, default=320, help="Width of the video to generate (if init image is not provided)")
    parser.add_argument("-HI", "--height", type=int, default=192, help="Height of the video (if init image is not provided)")
    parser.add_argument("-IW", "--image-width", type=int, default=1280, help="Width of the image to generate (if init image is not provided)")
    parser.add_argument("-IH", "--image-height", type=int, default=768, help="Height of the image (if init image is not provided)")
    parser.add_argument("-i", "--init-image", type=str, default=None, help="Path to initial image to use")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=32, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=30, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=20, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-IG", "--image-guidance-scale", type=float, default=12, help="Scale for guidance loss for 2d model (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-f", "--fps", type=int, default=16, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-t", "--times", type=int, default=4, help="How many times to continue to generate videos")
    parser.add_argument("-I", "--save-init", action="store_true", help="Save the init image to the output folder for reference")
    parser.add_argument("-N", "--include-model", action="store_true", help="Include the name of the model in the exported file")

    args = parser.parse_args()

    if args.read_prompts_from_file:
        prompt = read_prompts(args.prompts_file)[0]
    else:
        prompt = args.prompt

    conditioning_init_image = None
    if args.init_image is not None:
        conditioning_init_image = image_to_tensor(args.init_image)

    image_width = None
    if args.image_width is None:
        image_width = args.width
    else:
        image_width = args.image_width

    image_height = None
    if args.image_height is None:
        image_height = args.height
    else:
        image_height = args.image_height

    seed = 0
    if args.seed is None:
        seed = random.randint(0, 9999999)
    else:
        seed = args.seed

    videos = inference(
        model=args.model,
        prompt=args.prompt,
        read_prompts_from_file=args.read_prompts_from_file,
        prompts_file=args.prompts_file,
        prompts_interval=args.prompts_interval,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        image_width=image_width,
        image_height=image_height,
        model_2d=args.model_2d,
        num_frames=args.num_frames,
        conditioning_hidden_states=conditioning_init_image,
        min_conditioning_n_sample_frames=args.min_conditioning_n_sample_frames,
        max_conditioning_n_sample_frames=args.max_conditioning_n_sample_frames,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        device=args.device,
        seed=seed,
        xformers=args.xformers,
        sdp=args.sdp,
        times=args.times,
        save_init=args.save_init,
        output_dir=args.output_dir
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for video in [videos]:
        video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
        video = video.byte().cpu().numpy()

        video = enhance_contrast_clahe_4d(video)

        unique_id = str(uuid4())[:8]
        out_file = f"{args.output_dir}/{prompt}-{unique_id}.mp4"
        model_name = ""
        if args.include_model:
            model_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", os.path.basename(args.model))
        encoded_out_file = f"{args.output_dir}/{prompt}-{model_name}-{unique_id}_encoded.mp4"

        export_to_video(video, out_file, args.fps)

        try:
            encode_video(out_file, encoded_out_file, get_video_height(out_file))
            os.remove(out_file)
        except Exception as e:
            pass
