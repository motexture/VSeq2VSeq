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
import decord

from PIL import Image
from diffusers import TextToVideoSDPipeline, DiffusionPipeline
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Union
from einops import rearrange
from torch import Tensor
from tqdm import trange
from uuid import uuid4
from diffusers.utils import PIL_INTERPOLATION
from einops import rearrange
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from skimage import exposure

def match_histogram(frame, reference_image_path):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    matched = exposure.match_histograms(frame, reference_image, channel_axis=-1)
    return matched

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            for m in module:
                if isinstance(m, BasicTransformerBlock):
                    set_processors([m.attn1, m.attn2])
                    optim_count += 1

    print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def save_image(tensor, filename):
    tensor = tensor.cpu().numpy()
    tensor = tensor.transpose((1, 2, 0))
    tensor = (tensor * 255).astype('uint8')

    img = Image.fromarray(tensor)
    img.save(filename)

    return filename

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

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
        text_encoder=text_encoder.to(device=device),
        vae=vae.to(device=device),
        unet=unet.to(device=device),
    )

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
        latents = torch.randn(shape)

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
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device)
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
        latents_batch = latents[idx : idx + batch_size].to(pipe.device)
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

    noisy_latents = torch.randn(shape)
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
    video_diffusion_model: str,
    image_diffusion_model: str,
    prompt: str,
    read_prompts_from_file: bool,
    prompts_file: str,
    prompts_interval: int,
    negative_prompt: Optional[str] = None,
    video_width: int = 320,
    video_height: int = 192,
    image_width: int = None,
    image_height: int = None,
    num_frames: int = 16,
    num_conditioning_frames: int = 4,
    vae_batch_size: int = 32,
    num_steps: int = 50,
    guidance_scale: float = 20,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    times: int = 4,
    seed: Optional[int] = None,
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

    stable_diffusion_pipe = DiffusionPipeline.from_pretrained(image_diffusion_model).to(device)
    conditioning_hidden_states = stable_diffusion_pipe(prompt=prompt, negative_prompt=negative_prompt, width=image_width, height=image_height, guidance_scale=guidance_scale, output_type="pt").images[0]
    
    conditioning_hidden_states = conditioning_hidden_states.unsqueeze(0)
    conditioning_hidden_states = F.interpolate(conditioning_hidden_states, size=(video_height, video_width), mode='bilinear', align_corners=False)

    unique_id = str(uuid4())[:8]
    reference_image_path = save_image(conditioning_hidden_states.squeeze(0), f"{output_dir}/{prompt}-{unique_id}.png")

    conditioning_hidden_states = conditioning_hidden_states.unsqueeze(2)

    del stable_diffusion_pipe
    torch.cuda.empty_cache()

    with torch.autocast(device):
        pipe = initialize_pipeline(video_diffusion_model, device, xformers, sdp)

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
            conditioning_hidden_states = concatenated_latents[:, :, -num_conditioning_frames:, :, :]  

        video_latents = torch.cat(video_latents, dim=0)

        videos = decode(pipe, video_latents, vae_batch_size)

    return torch.cat(torch.unbind(videos, dim=0), dim=1), reference_image_path

if __name__ == "__main__":
    decord.bridge.set_bridge("torch")

    parser = argparse.ArgumentParser()
    parser.add_argument("-VD", "--video-diffusion-model", type=str, default="motexture/VSeq2VSeq", help="Path to video diffusion model")
    parser.add_argument("-ID", "--image-diffusion-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Path to image diffusion model")

    parser.add_argument("-p", "--prompt", type=str, default="A stormtrooper is surfing on the ocean", help="Text prompt to condition on")
    
    parser.add_argument("-RP", "--read-prompts-from-file", default=False, action="store_true", help="File path to prompts file")
    parser.add_argument("-PF", "--prompts-file", type=str, default=None, help="Path to text file with prompts")
    parser.add_argument("-PI", "--prompts-interval", type=int, default=4, help="Interval for switching prompts when reading from file. Prompts will be interpolated linearly on this interval")

    parser.add_argument("-NP", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    
    parser.add_argument("-FR", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-CF", "--num-conditioning-frames", type=int, default=4, help="Total number of frames to sample for conditioning")

    parser.add_argument("-VW", "--video-width", type=int, default=448, help="Width of the video to generate")
    parser.add_argument("-VH", "--video-height", type=int, default=256, help="Height of the video to generate")
    parser.add_argument("-IW", "--image-width", type=int, default=1280, help="Width of the image to generate")
    parser.add_argument("-IH", "--image-height", type=int, default=768, help="Height of the image")

    parser.add_argument("-VB", "--vae-batch-size", type=int, default=32, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage)")
    parser.add_argument("-NS", "--num-steps", type=int, default=50, help="Number of diffusion steps to run per frame")
    parser.add_argument("-GS", "--guidance-scale", type=float, default=12, help="Scale for guidance loss")
    parser.add_argument("-f", "--fps", type=int, default=16, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation")
    parser.add_argument("-s", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible")
    parser.add_argument("-t", "--times", type=int, default=4, help="How many times to continue to generate videos")
    parser.add_argument("-OD", "--output-dir", type=str, default="./output", help="Directory to save output video to")

    args = parser.parse_args()

    if args.read_prompts_from_file:
        prompt = read_prompts(args.prompts_file)[0]
    else:
        prompt = args.prompt

    seed = 0
    if args.seed is None:
        seed = random.randint(0, 9999999)
    else:
        seed = args.seed

    os.makedirs(args.output_dir, exist_ok=True)

    videos, reference_image_path = inference(
        video_diffusion_model=args.video_diffusion_model,
        image_diffusion_model=args.image_diffusion_model,
        prompt=args.prompt,
        read_prompts_from_file=args.read_prompts_from_file,
        prompts_file=args.prompts_file,
        prompts_interval=args.prompts_interval,
        negative_prompt=args.negative_prompt,
        video_width=args.video_width,
        video_height=args.video_height,
        image_width=args.image_width,
        image_height=args.image_height,
        num_frames=args.num_frames,
        num_conditioning_frames=args.num_conditioning_frames,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        seed=seed,
        xformers=args.xformers,
        sdp=args.sdp,
        times=args.times,
        output_dir=args.output_dir
    )

    for video in [videos]:
        video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
        video = video.byte().cpu().numpy()

        matched_frames = []
        for frame in video:
            matched_frame = match_histogram(frame, reference_image_path)
            matched_frames.append(matched_frame)

        unique_id = str(uuid4())[:8]
        out_file = f"{args.output_dir}/{prompt}-{unique_id}.mp4"
        model_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", os.path.basename(args.video_diffusion_model))
        encoded_out_file = f"{args.output_dir}/{prompt}-{model_name}-{unique_id}_encoded.mp4"

        export_to_video(matched_frames, out_file, args.fps)

        try:
            encode_video(out_file, encoded_out_file, get_video_height(out_file))
            os.remove(out_file)
        except Exception as e:
            pass
