import argparse
import os
import random
import string
import time
from typing import Dict, List, Sequence, Tuple
from types import SimpleNamespace
from accelerate import Accelerator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geometry_utils import (discover_font_files, load_font_glyph_polylines,
                            stack_and_normalize)
from models.latent_geometry import GlyphGeometryVAE


class GlyphPointDataset(Dataset):
    def __init__(self,
                 font_root: str,
                 chars: Sequence[str],
                 max_points: int = 512,
                 min_points: int = 32,
                 samples_per_curve: int = 32,
                 max_retries: int = 10):
        self.font_paths = discover_font_files(font_root)
        self.chars = chars
        self.samples: List[Tuple[str, str]] = [
            (font_path, ch) for font_path in self.font_paths for ch in self.chars
        ]
        if not self.samples:
            raise RuntimeError("No glyph samples discovered.")
        self.max_points = max_points
        self.min_points = min_points
        self.samples_per_curve = samples_per_curve
        self.max_retries = max_retries
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._units_cache: Dict[str, int] = {}

    def __len__(self):
        return len(self.samples)

    def _load_font(self, font_path: str):
        if font_path not in self._cache:
            glyph_map, units = load_font_glyph_polylines(
                font_path, self.chars, samples_per_curve=self.samples_per_curve)
            self._cache[font_path] = glyph_map
            self._units_cache[font_path] = units
        return self._cache[font_path], self._units_cache[font_path]

    def _prepare_points(self, polylines, units_per_em):
        points = stack_and_normalize(polylines, units_per_em)
        if len(points) < self.min_points:
            return None
        length = min(len(points), self.max_points)
        padded = np.zeros((self.max_points, 2), dtype=np.float32)
        mask = np.zeros((self.max_points,), dtype=np.float32)
        padded[:length] = points[:length]
        mask[:length] = 1.0
        return padded, mask, length

    def __getitem__(self, index):
        attempt_index = index
        for _ in range(self.max_retries):
            font_path, char = self.samples[attempt_index]
            glyph_map, units = self._load_font(font_path)
            polylines = glyph_map.get(char)
            if polylines:
                prepared = self._prepare_points(polylines, units)
                if prepared is not None:
                    points, mask, length = prepared
                    return {
                        "points": torch.from_numpy(points),
                        "mask": torch.from_numpy(mask),
                        "length": length,
                        "font_path": font_path,
                        "char": char,
                    }
            attempt_index = random.randint(0, len(self.samples) - 1)
        raise RuntimeError("Failed to sample a valid glyph after retries.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a latent glyph geometry VAE.")
    parser.add_argument("--font_root", required=True,
                        help="Directory containing font files.")
    parser.add_argument("--output_dir", default="outputs/latent_geometry",
                        help="Directory to store checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--min_points", type=int, default=32)
    parser.add_argument("--samples_per_curve", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed_precision",
                        type=str,
                        choices=["no", "fp16", "bf16"],
                        default="no")
    return parser.parse_args()


def _as_namespace(args):
    if isinstance(args, SimpleNamespace):
        return args
    if isinstance(args, dict):
        return SimpleNamespace(**args)
    return SimpleNamespace(**vars(args))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, epoch, step, model, optimizer):
    state = {
        "epoch": epoch,
        "global_step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def main(args=None):
    if args is None:
        args = parse_args()
    args = _as_namespace(args)
    set_seed(args.seed)
    mp_mode = getattr(args, "mixed_precision", "no") or "no"
    accelerator = Accelerator(
        cpu=getattr(args, "cpu", False),
        mixed_precision=mp_mode,
    )
    device = accelerator.device

    characters = string.ascii_letters + string.digits
    dataset = GlyphPointDataset(
        args.font_root,
        characters,
        max_points=args.max_points,
        min_points=args.min_points,
        samples_per_curve=args.samples_per_curve,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(accelerator.state.device_type == "cuda"),
    )

    model = GlyphGeometryVAE(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, device)
        accelerator.print(f"Resumed from {args.resume} at epoch {start_epoch}")

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    steps_per_epoch = len(dataloader)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            points = batch["points"].to(device)
            lengths = batch["length"].to(device)
            mask = batch["mask"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            with accelerator.autocast():
                recon, mu, logvar = model(points, lengths)
                mse = torch.sum(((recon - points)**2) * mask) / \
                    mask.sum().clamp(min=1.0)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                      logvar.exp()) / points.size(0)
                loss = mse + args.kl_weight * kl
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += mse.item()
            epoch_kl += kl.item()
            global_step += 1

            if (step + 1) % args.log_every == 0:
                accelerator.print(
                    f"[Epoch {epoch+1}/{args.epochs}] Step {step+1}/{steps_per_epoch} "
                    f"Loss: {loss.item():.4f} Recon: {mse.item():.4f} KL: {kl.item():.4f}")

        duration = time.time() - start_time
        accelerator.print(
            f"Epoch {epoch+1} | Loss {epoch_loss/steps_per_epoch:.4f} | Recon {epoch_recon/steps_per_epoch:.4f} | "
            f"KL {epoch_kl/steps_per_epoch:.4f} | {duration:.1f}s")

        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, f"latent_geometry_epoch_{epoch+1}.pt")
            if accelerator.is_main_process:
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(ckpt_path, epoch + 1, global_step, unwrapped, optimizer)
                save_checkpoint(os.path.join(args.output_dir, "latest.pt"),
                                epoch + 1, global_step, unwrapped, optimizer)


if __name__ == "__main__":
    main(parse_args())

