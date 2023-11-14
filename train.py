# Adapted from https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/main/train.py

import argparse
import datetime
import inspect
import math
import os
import gc
import shutil
import json
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
import subprocess
import wandb
import random
import numpy as np

from tqdm import trange
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from typing import Dict, List
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange
from utils.dataset import VideoFolderDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, DiffusionPipeline
from pipeline.pipeline import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0

def average_contrast(video):
    num_frames = video.shape[0]
    avg_contrast = 0

    for f in range(num_frames):
        frame = cv2.cvtColor(video[f], cv2.COLOR_BGR2GRAY)
        avg_contrast += frame.std()

    avg_contrast /= num_frames

    output_video = np.zeros_like(video)
    for f in range(num_frames):
        frame = video[f]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_contrast = gray_frame.std()
        
        alpha = avg_contrast / (current_contrast + 1e-6)
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        
        output_video[f] = adjusted_frame

    return output_video

def enhance_contrast_clahe_4d(tensor, clip_limit=1.2, tile_grid_size=(1,1), gamma=0.98):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    output = np.empty_like(tensor)

    for f in range(tensor.shape[0]):
        for c in range(tensor.shape[3]):
            enhanced_frame = clahe.apply(tensor[f, :, :, c])
            enhanced_frame = np.power(enhanced_frame, gamma)
            enhanced_frame = np.clip(enhanced_frame, 0, 255)
            output[f, :, :, c] = enhanced_frame

    return output

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_image(tensor, filename):
    tensor = tensor.cpu().numpy()
    tensor = tensor.transpose((1, 2, 0))
    tensor = (tensor * 255).astype('uint8')
    img = Image.fromarray(tensor)
    img.save(filename)

def decode(pipe: TextToVideoSDPipeline, latents: torch.Tensor, batch_size: int = 32):
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

def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )
    
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]
    return images

def tensor2img(image: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], file_name='image.png') -> np.ndarray:
    mean = torch.tensor(mean, device=image.device).reshape(1, -1, 1, 1)
    std = torch.tensor(std, device=image.device).reshape(1, -1, 1, 1)
    
    image = image.mul_(std).add_(mean)
    image.clamp_(0, 1)
    
    b, c, h, w = image.shape
    image = image.permute(0, 2, 3, 1).reshape(b * h, w, c)
    image = (image.cpu().numpy() * 255).astype("uint8")

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(file_name, image)

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

def read_deepspeed_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def handle_trainable_modules(model, trainable_modules=None, is_enabled=True):
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break
                    
                if tm in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params +=1

    if unfrozen_params > 0:
        print(f"{unfrozen_params} params have been unfrozen for training.")

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

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    return out_dir

def load_primary_models(pretrained_model_path, upgrade_model=False):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    if upgrade_model:
        unet = UNet3DConditionModel.from_pretrained_3d(pretrained_model_path, subfolder="unet")
    else:
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    print(f'The model has {count_parameters(unet):,} trainable parameters')

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")

    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def sample_noise(latents, noise_strength, use_offset_noise):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def should_save(global_step, checkpointing_steps):
    return global_step % checkpointing_steps == 0

def should_validate(global_step, validation_steps):
    return global_step % validation_steps == 0

def should_sample(global_step, sample_steps):
    return global_step % sample_steps == 0

def save_pipe(
        path, 
        global_step,
        unet, 
        vae, 
        text_encoder, 
        tokenizer,
        scheduler,
        output_dir,
        is_checkpoint=False,
        remove_older_checkpoints=False
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        if remove_older_checkpoints:
            existing_checkpoints = [d for d in os.listdir(output_dir) if 'checkpoint-' in d]
            existing_checkpoints = sorted(existing_checkpoints, key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))

            while len(existing_checkpoints) > 1:
                shutil.rmtree(os.path.join(output_dir, existing_checkpoints.pop(0)))
    else:
        save_path = output_dir

    pipeline = TextToVideoSDPipeline.from_pretrained(path, 
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler)
    
    pipeline.save_pretrained(save_path)

    print(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline

    torch.cuda.empty_cache()
    gc.collect()

def main(
    pretrained_3d_model_path: str,
    pretrained_2d_model_path: str,
    upgrade_model: bool,
    output_dir: str,
    train_data: Dict,
    sample_data: Dict,
    epochs: int = 1,
    train_dataset_size: float = 0.99,
    checkpointing_steps: int = 100,
    validation_steps: int = 100,
    sample_steps: int = 100,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    use_offset_noise: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
    enable_torch_2_attn: bool = True,
    offset_noise_strength: float = 0.1,
    resume_from_checkpoint: bool = False,
    save_only_best: bool = True,
    resume_step: int = None,
    **kwargs
):
    accelerator = Accelerator()

    *_, config = inspect.getargvalues(inspect.currentframe())

    set_seed(seed)

    if accelerator.is_local_main_process:
        wandb.init(
            project=output_dir.split('/')[-1],
            config={
                "seed": seed,
                "learning_rate": train_data.learning_rate,
                "weight_decay": train_data.weight_decay,
                "batch_size_per_gpu": train_data.batch_size_per_gpu,
                "trainable_modules": ",".join(train_data.trainable_modules),
                "resolution": f"{train_data.width}x{train_data.height}",
                "frame_step": train_data.frame_step,
                "n_sample_frames": train_data.n_sample_frames,
                "epochs": epochs
            }
        )
        
        output_dir = create_output_folders(output_dir, config)
        
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_3d_model_path, upgrade_model)

    optimizer = AdamW(unet.parameters(), lr=train_data.learning_rate, betas=train_data.betas, eps=train_data.eps, weight_decay=train_data.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=train_data.cosine_annealing_t_max)

    pipe = DiffusionPipeline.from_pretrained(pretrained_2d_model_path, torch_dtype=torch.float16)
    
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    pipe = pipe.to(accelerator.device)

    freeze_models([text_encoder, vae, unet])

    vae.enable_slicing()

    train_dataset = VideoFolderDataset(**train_data, tokenizer=tokenizer, device=accelerator.device)

    train_size = int(train_dataset_size * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_data.batch_size_per_gpu,
        shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_data.batch_size_per_gpu,
        shuffle=False
    )

    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(unet, optimizer, train_dataloader, val_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    global_step = 0

    unet.train()

    if gradient_checkpointing:
        unet._set_gradient_checkpointing(value=True)

    handle_trainable_modules(unet, train_data.trainable_modules, is_enabled=True)

    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    best_validation_loss = 1000000
    
    progress_bar = tqdm(range(global_step, num_update_steps_per_epoch * epochs))
    progress_bar.set_description("Steps")

    def predict(batch):
        pixel_values = batch["pixel_values"]
        pixel_values = pixel_values.to(accelerator.device)

        latents = tensor_to_vae_latent(pixel_values, vae)

        random_slice = random.randint(train_data.min_conditioning_n_sample_frames, train_data.max_conditioning_n_sample_frames)

        conditioning_latents = latents[:, :, :random_slice, :, :]
        latents = latents[:, :, random_slice:random_slice + train_data.n_max_frames, :, :]
        
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        token_ids = batch['prompt_ids'].to(accelerator.device)
        encoder_hidden_states = text_encoder(token_ids)[0].detach()

        if noise_scheduler.prediction_type == "epsilon":
            noisy_target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            noisy_target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        noisy_latents.requires_grad = True
        conditioning_latents.requires_grad = True

        noise_pred = unet(noisy_latents, conditioning_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noisy_target.float(), reduction="mean")

        return loss
    
    for _ in range(0, epochs):
        for batch in train_dataloader:
            if resume_from_checkpoint and global_step < resume_step:
                progress_bar.update(1)
                global_step += 1
                continue

            with accelerator.accumulate(unet):                
                with accelerator.autocast():
                    loss = predict(batch)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad(set_to_none=True)

                if accelerator.is_local_main_process:
                    wandb.log({'Training Loss': loss.item()}, step=global_step)
                    wandb.log({'Learning Rate': optimizer.param_groups[0]['lr']}, step=global_step)

                    progress_bar.update(1)
                    global_step += 1

                    if should_save(global_step, checkpointing_steps):
                        save_pipe(
                            pretrained_3d_model_path,
                            global_step,
                            unet,
                            vae,
                            text_encoder,
                            tokenizer,
                            noise_scheduler,
                            output_dir,
                            is_checkpoint=True,
                            remove_older_checkpoints=False
                        )
                            
                    if should_sample(global_step, sample_steps):
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=False)
                        unet.eval()
                        
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            pretrained_3d_model_path,
                            text_encoder=text_encoder,
                            vae=vae,
                            unet=unet,
                            scheduler=noise_scheduler
                        )

                        prompt = batch["text_prompt"][0] if len(sample_data.prompt) <= 0 else sample_data.prompt

                        save_filename = f"{global_step}-{prompt}"
                        image_file = f"{output_dir}/samples/{save_filename}.png"
                        video_file = f"{output_dir}/samples/{save_filename}.mp4"
                        encoded_video_file = f"{output_dir}/samples/{save_filename}_encoded.mp4"
                        
                        conditioning_hidden_states = pipe(prompt, width=sample_data.image_width, height=sample_data.image_height, output_type="pt", generator=torch.Generator().manual_seed(seed)).images[0]
                        conditioning_hidden_states = F.interpolate(conditioning_hidden_states.unsqueeze(0), size=(sample_data.height, sample_data.width), mode='bilinear', align_corners=False).squeeze(0)
                        
                        save_image(conditioning_hidden_states, image_file)

                        conditioning_hidden_states = conditioning_hidden_states.unsqueeze(0).unsqueeze(2)

                        with accelerator.autocast():
                            with torch.no_grad():
                                video_latents = []
                                for t in range(0, sample_data.times):
                                    latents = pipeline(
                                        prompt=prompt,
                                        encode_to_latent=True if t == 0 else False,
                                        width=sample_data.width,
                                        height=sample_data.height,
                                        conditioning_hidden_states=conditioning_hidden_states,
                                        num_frames=sample_data.num_frames,
                                        num_inference_steps=sample_data.num_inference_steps,
                                        guidance_scale=sample_data.guidance_scale,
                                        output_type="latent",
                                        generator=torch.Generator().manual_seed(seed)
                                    ).frames
                                    
                                    video_latents.append(latents)
                                    
                                    random_slice = random.randint(train_data.min_conditioning_n_sample_frames, train_data.max_conditioning_n_sample_frames)
                                    conditioning_hidden_states = latents[:, :, -random_slice:, :, :]                                
                        
                        video_latents = torch.cat(video_latents, dim=0)

                        video_frames = decode(pipeline, video_latents)
                        video_frames = torch.cat(torch.unbind(video_frames, dim=0), dim=1)

                        video_frames = rearrange(video_frames, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
                        video_frames = video_frames.byte().cpu().numpy()

                        #video_frames = average_contrast(video_frames)
                        video_frames = enhance_contrast_clahe_4d(video_frames)

                        export_to_video(video_frames, video_file, sample_data.fps)

                        try:
                            encode_video(video_file, encoded_video_file, get_video_height(video_file))
                            os.remove(video_file)
                        except:
                            pass

                        del pipeline, video_frames
                        
                        torch.cuda.empty_cache()
                            
                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=True)
                        unet.train()

                    if should_validate(global_step, validation_steps):
                        validation_progress_bar = tqdm(range(0, len(val_dataloader)))
                        validation_progress_bar.set_description("Validation Steps")

                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=False)
                        unet.eval()

                        total_val_loss = 0.0

                        with accelerator.autocast():
                            with torch.no_grad():
                                for val_batch in val_dataloader:
                                    val_loss = predict(val_batch)
                                    total_val_loss += val_loss.item()

                                    validation_progress_bar.update(1)

                        average_val_loss = total_val_loss / len(val_dataloader)
                        wandb.log({'Validation Loss': average_val_loss}, step=global_step)

                        print(f"Validation loss: {average_val_loss} Best validation loss: {best_validation_loss}\n")
                        if average_val_loss < best_validation_loss:
                            best_validation_loss = average_val_loss

                            if save_only_best:
                                save_pipe(
                                    pretrained_3d_model_path,
                                    global_step,
                                    unet,
                                    vae,
                                    text_encoder,
                                    tokenizer,
                                    noise_scheduler,
                                    output_dir,
                                    is_checkpoint=True,
                                    remove_older_checkpoints=True
                                )

                        if gradient_checkpointing:
                            unet._set_gradient_checkpointing(value=True)
                        unet.train()

                        validation_progress_bar.close()
                        del validation_progress_bar

    if accelerator.is_local_main_process:
        save_pipe(
            pretrained_3d_model_path,
            global_step,
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            output_dir,
            is_checkpoint=False,
            remove_older_checkpoints=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training.yaml")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank of this process. Used for distributed training.')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))