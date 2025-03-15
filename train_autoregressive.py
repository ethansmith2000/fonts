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
from models.encoder import Encoder


import torch
from utils import detect_boxes, font_supports_all_chars, render_char
import os
import string
import matplotlib.font_manager as fm
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from glyph import load_transformer, visualize_commands, path_to_string, string_to_path, normalize_commands, open_font, process_all_glyphs, is_valid_font
import glob
import random

class FontSVGDataset(torch.utils.data.Dataset):
    def __init__(self, 
                # font_dir='/home/ubuntu/fonts/scraper/fonts_unpacked', 
                font_dir='/home/ubuntu/fonts/newfonts/newfonts',
                num_glyphs=7
                ):
        self.paths = glob.glob(os.path.join(font_dir, "**", "*.ttf"), recursive=True)
        print("found", len(self.paths), "fonts")
        self.num_glyphs = num_glyphs
        self.num_map = {
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "0": "zero",
        }

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        for i in range(10):
            font = open_font(self.paths[idx])
            # is_valid, error_message = is_valid_font(font)
            # if not is_valid:
            #     # print(error_message)
            #     idx = random.randint(0, len(self.paths) - 1)
            #     continue
            all_glyph_commands = process_all_glyphs(font)
            if all_glyph_commands is None:
                idx = random.randint(0, len(self.paths) - 1)
                continue

            font_name = self.paths[idx].split("/")[-1].split(".")[0]

            # all_glyphs = random.sample(list(all_glyph_commands.keys()), self.num_glyphs)

            rand_start_idx = random.randint(0, len(all_glyph_commands) - self.num_glyphs)
            all_glyphs = list(all_glyph_commands.keys())[rand_start_idx:rand_start_idx + self.num_glyphs]

            string = ""
            for i, char in enumerate(all_glyphs):
                char_token = self.num_map[char] if char in self.num_map else char
                string += " " + char_token + " "
                commands = normalize_commands(all_glyph_commands[char])
                string += path_to_string(commands)
                if i < len(all_glyphs) - 1:
                    string += "[SEP] "
            
            return (string, font_name)
        else:
            raise ValueError("Tokenized is None")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize dataset and dataloader
    with TimerWithMessage("Loading dataset...", "Dataset loading"):
        dataset = FontSVGDataset(num_glyphs=args["num_glyphs"])
    collate_fn = lambda x: x
    dataloader = DataLoader(dataset, 
                            batch_size=args["batch_size"], 
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=True,
                            collate_fn=collate_fn,
                            persistent_workers=True
                            )

    tokenizer, transformer = load_transformer(pos_emb_len=args["pos_emb_len"])

    if args["use_wandb"]:
        wandb.init(project="fonts-autoregressive", config=args)
    
    # Initialize model and optimizer
    model = transformer.to(device).requires_grad_(False)
    model.gradient_checkpointing_enable()

    params_to_train = [p for n, p in model.named_parameters() if "wte" in n or "lm_head" in n or "wpe" in n] if args["freeze_backbone"] else [p for p in model.parameters()]
    
    for p in params_to_train:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(params_to_train, lr=args["learning_rate"], betas=args["betas"], weight_decay=args["weight_decay"], fused=True)

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

    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_lambda(step, args["warmup_steps"], total_steps)
    )


    # mixed precision
    mp_dtype = torch.float16 if args["mixed_precision"] == "fp16" else torch.bfloat16
    mp_enabled = args["mixed_precision"] is not None
    scaler = torch.GradScaler(enabled=args["mixed_precision"] == "fp16")

    # load checkpoint
    global_step = 0
    if args["load_checkpoint"] is not None:
        checkpoint = torch.load(args["load_checkpoint"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint from {args['load_checkpoint']}")


    # compiling
    if args["compile_optimizer"]:
        # optimizer.step = torch.compile(optimizer.step)
        scaler.step = torch.compile(scaler.step)
        optimizer.zero_grad = torch.compile(optimizer.zero_grad)
        grad_clip = torch.compile(torch.nn.utils.clip_grad_norm_)
    else:
        grad_clip = torch.nn.utils.clip_grad_norm_
    
    if args["compiled"]:
        model.forward = torch.compile(model.forward, dynamic=args["compile_dynamic"])


    times = {
        "fwd": 0,
        "bwd": 0,
        "data": 0,
        "tok": 0,
        "opt": 0,
    }

    # Training loop
    tik = None
    for epoch in range(args["epochs"]):
        model.train()        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        for batch in pbar:
            tok = time.time()
            if tik is not None:
                times["data"] = tok - tik

            glyphs = [b[0] for b in batch]
            font_names = [b[1] for b in batch]

            # filter out sequences that are too long
            font_names = [f for g, f in zip(glyphs, font_names) if len(g) <= args["max_length"]]
            glyphs = [g for g in glyphs if len(g) <= args["max_length"]]

            # tokenize
            with Timer("tok", times):
                batch = tokenizer(glyphs, return_tensors="pt", padding="longest").input_ids.to(device)

            inps = batch[:, :-1]
            targs = batch[:, 1:]

            if inps.shape[1] > args["pos_emb_len"]:
                raise ValueError("Input too long")

            # Forward pass
            with torch.autocast("cuda", dtype=mp_dtype, enabled=mp_enabled):
                with Timer("fwd", times):
                    out = model(inps, labels=targs)
                    loss = out.loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            with Timer("bwd", times):
                scaler.scale(loss).backward()

            # # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            with Timer("opt", times):
                grad_norm = 0
                if args["max_norm"] is not None:
                    grad_norm = grad_clip(model.parameters(), args["max_norm"])

                scaler.step(optimizer)
                scaler.update()

                # scheduler step
                scheduler.step()

            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], grd=grad_norm.item(), **times)
            tik = time.time()

            if args["use_wandb"]:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    **times
                })

            if global_step % args["log_images_every"] == 0:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=mp_dtype, enabled=mp_enabled):
                        char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"
                        start_idx = random.randint(0, len(char_list) - args["num_glyphs"])
                        char = char_list[start_idx]
                        toks = tokenizer(" " + char, return_tensors="pt").input_ids.to(device)
                        toks = toks.repeat(args["sample_batch_size"], 1)
                        out = model.generate(toks, max_new_tokens=500)
                        out = out.detach().cpu().numpy()
                        out = tokenizer.batch_decode(out, skip_special_tokens=True)

                        # import pdb; pdb.set_trace()

                        # for each sequence, just get the first letter, trim at [SEP]
                        out = [o.split("[SEP]")[0] for o in out if "[SEP]" in o]

                        if len(out) > 0:
                            imgs = []
                            for o in out:
                                img = visualize_commands(string_to_path(o), show=False)
                                imgs.append(img)
                            wandb.log({"images": imgs})


            global_step += 1

            # save checkpoint
            if global_step % args["save_every"] == 0:
                if not os.path.exists("checkpoints"):   
                    os.makedirs("checkpoints")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if args["save_optimizer"] else None,
                }, f'checkpoints/step_{global_step}.pt')
        

if __name__ == "__main__":
    #tf32 enable, matmul precision high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    args = dict(
        epochs=300,
        warmup_steps=100,
        batch_size=60,
        learning_rate=2.0e-4,
        weight_decay=0.01,
        betas=(0.92, 0.989),
        max_norm=1.0,
        freeze_backbone=True,
        mixed_precision="fp16",

        use_wandb=True,
        log_images_every=100,
        sample_batch_size=16,
        save_every=1000,
        save_optimizer=False,

        load_checkpoint=None,
        # load_checkpoint="/home/ubuntu/fonts/checkpoints/step_10000.pt",
        compile_optimizer=True,
        compiled=False,
        compile_dynamic=True,
        max_length=11_500,
        num_glyphs=7,
        pos_emb_len=3072,
    )
    train(args)
