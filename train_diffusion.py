import torch
from utils import detect_boxes, font_supports_all_chars, render_char, TimerWithMessage, Timer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader_pre_rendered import FontImageDataset
from dataloader import FontDataset
import time 
import wandb 
import numpy as np
from PIL import Image
import diffusers
from diffusers import UNet2DConditionModel
from torch import nn
from diffusers import FlowMatchEulerDiscreteScheduler
from typing import Optional, Union
import math 
from copy import deepcopy
import os
from accelerate import Accelerator

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def compute_density_for_timestep_sampling(
    batch_size: int,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

def get_sigmas(timesteps, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def get_model():
    block_out_channels=(16, # 512x512
                        32, # 256x256
                        64, # 128x128
                        128, # 64x64
                        256, # 32x32
                        512, # 16x16
                        1024, # 8x8
                        )
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    )
    up_block_types=(
        "AttnUpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D",
        "UpBlock2D",
    )

    unet = UNet2DConditionModel(block_out_channels=block_out_channels,
                                out_channels=1, 
                                in_channels=1, 
                                up_block_types=up_block_types, 
                                down_block_types=down_block_types, 
                                mid_block_type="UNetMidBlock2D",
                                resnet_time_scale_shift="scale_shift",
                                # time_cond_proj_dim=64,
                                norm_num_groups=16,
                                layers_per_block=1,
                                addition_embed_type="image",
                                encoder_hid_dim=64,
                                )

    condition_mlp = nn.Sequential(
        nn.Linear(64, 128, bias=True),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 64, bias=True),
    )
    for name, param in condition_mlp.named_parameters():
        if len(param.shape) > 1:
            #xavier_uniform_
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    unet.register_module("condition_mlp", condition_mlp)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total parameters: {total_params}")
    return unet


def train(args):
    mp_mode = args.get("mixed_precision") or "no"
    accelerator = Accelerator(mixed_precision=mp_mode)
    device = accelerator.device
    
    # Initialize dataset and dataloader
    dataset_cls = FontImageDataset if args["pre_rendered"] else FontDataset
    with TimerWithMessage("Loading dataset...", "Dataset loading"):
        dataset = dataset_cls()
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=10, pin_memory=True)

    use_wandb = args["use_wandb"] and accelerator.is_main_process
    if use_wandb:
        wandb.init(project="fonts-vae", config=args)
    
    # Initialize model and optimizer
    model = get_model().to(device)
    model.enable_gradient_checkpointing()
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        "black-forest-labs/FLUX.1-dev", subfolder="scheduler", use_dynamic_shifting = False
    )
    noise_scheduler_copy = deepcopy(noise_scheduler)
    if args["compiled"]:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], betas=args["betas"], weight_decay=args["weight_decay"])

    # Create a learning rate scheduler with linear warmup and decay
    def get_lr_lambda(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay phase
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    # Calculate total steps based on epochs and batch size
    total_steps = len(dataloader) * args["epochs"]
    warmup_steps = 100
    
    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_lambda(step, warmup_steps, total_steps)
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    times = {
        "forward": 0,
        "backward": 0,
        "data_load": 0,
    }

    global_step = 0
    
    dummy_enc_states = torch.randn(1, 1, 64).to(device)
    
    # Training loop
    tik = None
    for epoch in range(args["epochs"]):
        model.train()        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            tok = time.time()
            if tik is not None:
                times["data_load"] = tok - tik
            images = batch["image"].to(device)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(images)
            bsz = images.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                # weighting_scheme="logit_normal",
                weighting_scheme=None,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=images.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, noise_scheduler_copy, n_dim=images.ndim, dtype=images.dtype)
            noisy_model_input = (1.0 - sigmas) * images + sigmas * noise

            target = noise - images

            conditions = torch.randn(bsz, 64).to(device)
            
            # Forward pass
            with accelerator.autocast():
                with Timer("forward", times):
                    conditions = model.condition_mlp(conditions)
                    pred = model(noisy_model_input, 
                                timesteps/1000, 
                                encoder_hidden_states=dummy_enc_states, 
                                # timestep_cond=conditions, 
                                return_dict=False,
                                added_cond_kwargs={"image_embeds": conditions}
                                )[0]
            
            # loss
            # loss = F.mse_loss(pred.float(), target.float(), reduction='mean')

            loss_weights = compute_loss_weighting_for_sd3(weighting_scheme="cosmap", sigmas=sigmas)
            loss = ((pred.float() - target.float())**2).reshape(bsz, -1).mean(dim=-1) * loss_weights
            loss = loss.mean()


            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            with Timer("backward", times):
                accelerator.backward(loss)

            optimizer.step()

            # scheduler step
            scheduler.step()

            if accelerator.is_local_main_process:
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], **times)
            tik = time.time()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    **times
                })
            if (global_step % args["log_images_every"] == 0) and accelerator.is_main_process:
                logging_model = accelerator.unwrap_model(model)
                with torch.no_grad():
                    with accelerator.autocast():
                        latents = torch.randn(args["sample_batch_size"], 1, 512, 512).to(device)
                        conditions = torch.randn(args["sample_batch_size"], 64).to(device)
                        conditions = logging_model.condition_mlp(conditions)

                        noise_scheduler.set_timesteps(args["diffusion_steps"])
                        for i, t in enumerate(noise_scheduler.timesteps):
                            pred = logging_model(latents, 
                                                 t/1000, 
                                                 encoder_hidden_states=dummy_enc_states, 
                                                 # timestep_cond=conditions, 
                                                 return_dict=False,
                                                 added_cond_kwargs={"image_embeds": conditions}
                                                 )[0]
                            latents = noise_scheduler.step(pred, t, latents).prev_sample
                        
                        images = latents.float().detach().cpu().numpy().squeeze(1)
                        images = (images * 127.5 + 127.5).astype(np.uint8)
                        images = [wandb.Image(Image.fromarray(images[i])) for i in range(images.shape[0])]
                        if use_wandb:
                            wandb.log({"images": images})

            global_step += 1
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            accelerator.wait_for_everyone()
            if not os.path.exists("checkpoints"):   
                os.makedirs("checkpoints")
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/epoch_{epoch+1}.pt')


if __name__ == "__main__":
    #tf32 enable, matmul precision high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    args = dict(
        epochs=300,
        batch_size=60,
        learning_rate=2.0e-4,
        weight_decay=0.001,
        betas=(0.92, 0.989),
        pre_rendered=True,
        use_wandb=True,
        log_images_every=100,
        compiled=True,
        sample_batch_size=16,
        diffusion_steps=100,
        mixed_precision="fp16"
    )
    train(args)
