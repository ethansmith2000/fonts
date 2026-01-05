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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from geometry_utils import (bitmap_to_sdf, discover_font_files,
                            render_glyph_bitmap, sample_sdf_points)
from models.implicit_sdf import ImplicitGlyphSDF


class GlyphSDFDataset(Dataset):
    def __init__(self,
                 font_root: str,
                 chars: Sequence[str],
                 image_size: int = 256,
                 num_samples: int = 4096,
                 max_retries: int = 10):
        self.font_paths = discover_font_files(font_root)
        self.chars = chars
        self.samples: List[Tuple[str, str]] = [
            (fp, ch) for fp in self.font_paths for ch in self.chars
        ]
        if not self.samples:
            raise RuntimeError("No glyph samples discovered.")
        self.image_size = image_size
        self.num_samples = num_samples
        self.max_retries = max_retries
        self._sdf_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.samples)

    def _load_sdf(self, font_path: str, char: str):
        key = (font_path, char)
        if key not in self._sdf_cache:
            bitmap = render_glyph_bitmap(
                font_path, char, image_size=self.image_size)
            sdf = bitmap_to_sdf(bitmap) if bitmap is not None else None
            self._sdf_cache[key] = sdf
        return self._sdf_cache[key]

    def __getitem__(self, index):
        attempt_index = index
        for _ in range(self.max_retries):
            font_path, char = self.samples[attempt_index]
            sdf_map = self._load_sdf(font_path, char)
            if sdf_map is not None:
                coords, sdf = sample_sdf_points(
                    sdf_map, self.num_samples, rng=self._rng)
                return {
                    "coords": torch.from_numpy(coords),
                    "sdf": torch.from_numpy(sdf).unsqueeze(-1),
                    "font_path": font_path,
                    "char": char,
                }
            attempt_index = random.randint(0, len(self.samples) - 1)
        raise RuntimeError("Failed to sample glyph SDF after retries.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural implicit glyph SDF model.")
    parser.add_argument("--font_root", required=True,
                        help="Directory containing font files.")
    parser.add_argument("--output_dir", default="outputs/implicit_sdf",
                        help="Directory to store checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--fourier_features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--eikonal_weight", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=17)
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
    dataset = GlyphSDFDataset(
        args.font_root,
        characters,
        image_size=args.image_size,
        num_samples=args.num_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(accelerator.state.device_type == "cuda"),
    )

    model = ImplicitGlyphSDF(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fourier_frequencies=args.fourier_features,
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
        epoch_eik = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            coords = batch["coords"].to(device)
            sdf = batch["sdf"].to(device)
            coords = coords.view(-1, 2).contiguous()
            sdf = sdf.view(-1, 1)

            requires_eikonal = args.eikonal_weight > 0
            coords.requires_grad_(requires_eikonal)

            optimizer.zero_grad()
            with accelerator.autocast():
                preds = model(coords)
                recon_loss = F.l1_loss(preds, sdf)
            if requires_eikonal:
                grads = torch.autograd.grad(
                    outputs=preds,
                    inputs=coords,
                    grad_outputs=torch.ones_like(preds),
                    create_graph=True,
                )[0]
                eikonal_loss = ((grads.norm(dim=-1) - 1.0)**2).mean()
            else:
                eikonal_loss = torch.tensor(0.0, device=device)

            loss = recon_loss + args.eikonal_weight * eikonal_loss
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_eik += eikonal_loss.item()
            global_step += 1

            if (step + 1) % args.log_every == 0:
                accelerator.print(
                    f"[Epoch {epoch+1}/{args.epochs}] Step {step+1}/{steps_per_epoch} "
                    f"Loss: {loss.item():.4f} Recon: {recon_loss.item():.4f} "
                    f"Eik: {eikonal_loss.item():.4f}")

        duration = time.time() - start_time
        accelerator.print(
            f"Epoch {epoch+1} | Loss {epoch_loss/steps_per_epoch:.4f} | Recon {epoch_recon/steps_per_epoch:.4f} | "
            f"Eik {epoch_eik/steps_per_epoch:.4f} | {duration:.1f}s")

        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, f"implicit_sdf_epoch_{epoch+1}.pt")
            if accelerator.is_main_process:
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(ckpt_path, epoch + 1, global_step, unwrapped, optimizer)
                save_checkpoint(os.path.join(args.output_dir, "latest.pt"),
                                epoch + 1, global_step, unwrapped, optimizer)


if __name__ == "__main__":
    main(parse_args())

