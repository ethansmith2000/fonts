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
from models.vae import VAE
from accelerate import Accelerator


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
    model = VAE().to(device)
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
            
            # Forward pass
            with accelerator.autocast():
                with Timer("forward", times):
                    recon_images, mu, log_var = model(images)
            
            # loss
            recon = F.mse_loss(recon_images, images, reduction='mean')
            kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon + kl * args["kl_weight"]
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            with Timer("backward", times):
                accelerator.backward(loss)
            optimizer.step()

            # scheduler step
            scheduler.step()

            if accelerator.is_local_main_process:
                pbar.set_postfix(loss=loss.item(), recon=recon.item(), kl=kl.item(), **times)
            tik = time.time()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "recon": recon.item(),
                    "kl": kl.item(),
                    "lr": scheduler.get_last_lr()[0],
                    **times
                })

            if global_step % args["log_images_every"] == 0 and accelerator.is_main_process:
                # Convert from [-1,1] to [0,255] range and adjust dimensions for wandb
                # Shape is (batch, channels, height, width) but wandb expects (batch, height, width, channels)
                recon_np = (recon_images.squeeze(1).detach().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
                images_np = (images.squeeze(1).detach().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)

                recon_pil = [Image.fromarray(recon_np[i]) for i in range(recon_np.shape[0])]
                images_pil = [Image.fromarray(images_np[i]) for i in range(images_np.shape[0])]
                
                if use_wandb:
                    wandb.log({
                        "recon_images": [wandb.Image(img) for img in recon_pil],
                        "original_images": [wandb.Image(img) for img in images_pil],
                    })
            global_step += 1
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'vae_checkpoint_epoch_{epoch+1}.pt')


if __name__ == "__main__":
    #tf32 enable, matmul precision high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    args = dict(
        epochs=100,
        batch_size=64,
        learning_rate=2.5e-4,
        weight_decay=0.001,
        betas=(0.9, 0.999),
        kl_weight=0.01,
        pre_rendered=True,
        use_wandb=True,
        log_images_every=100,
        compiled=True,
        mixed_precision="fp16",
    )
    train(args)
