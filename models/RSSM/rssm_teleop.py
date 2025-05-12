"""
Tele‑op FPV simulator for a trained RSSM world‑model
====================================================
* Streams the model’s imagined camera view in real‑time
* Reads game‑pad (or keyboard) inputs via pygame and feeds them as twist commands
* Runs on CUDA, Apple MPS, or CPU automatically

Usage
-----
$ python rssm_teleop.py \
      --checkpoint rssm_20250511_123456_best.pth \
      --img-size 64                      # must match training

Controls
~~~~~~~~
Game‑pad (Xbox / PS style):
  LX  : forward/back (+Y) and strafe (‑X)
  LT/RT (or RX) : rotate (theta)

Keyboard fallback:
  W/S : +Y / ‑Y   (forward / back)
  A/D : ‑X / +X   (left / right)
  Q/E : +theta / ‑theta (ccw / cw)
  ESC : quit

Notes
~~~~~
* We roll out with the **prior mean** (no sampling) for smoother visuals.
* The first frame is the model’s unconditional prior; you can optionally
  pass `--seed-image path.jpg` to start from a real frame.
"""

import argparse, time, sys, math, os
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

# ===== import model code (assumes rssm_world_model.py is on PYTHONPATH) =====
from rssm_world_model import WorldModel, LATENT, HIDDEN, ImageEncoder, ImageDecoder, RSSM

# ---------------------------- helpers --------------------------------------

def choose_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_seed_image(path, img_size, device):
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img = tf(Image.open(path).convert('RGB')).unsqueeze(0).to(device)  # [1,3,H,W]
    return img

# ---------------------------- main ----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run trained RSSM interactively")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--seed-image', type=str, default=None,
                        help='Optional path to initial observation frame')
    parser.add_argument('--force-cpu', action='store_true')
    args = parser.parse_args()

    device = choose_device(args.force_cpu)
    print(f"Using device: {device}")

    # --- build model and load weights (decoder required for rendering) ---
    model = WorldModel(args.img_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    rssm = model.rssm  # shortcut
    dec  = model.dec
    enc  = model.enc

    # --- pygame setup ---
    pygame.init()
    size = args.img_size*4  # upscale 4× for display
    screen = pygame.display.set_mode((size,size))
    pygame.display.set_caption('RSSM Tele‑op FPV')
    clock = pygame.time.Clock()

    # game‑pad (optional)
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print('Game‑pad detected:', joystick.get_name())
    else:
        print('No game‑pad detected, using keyboard controls')

    # --- initial latent state ---
    h = torch.zeros(1, HIDDEN, device=device)
    if args.seed_image:
        obs0 = load_seed_image(args.seed_image, args.img_size, device)
        mu, lv = enc(obs0)
        z = mu  # deterministic seed
        h = rssm.rnn(torch.cat([z, torch.zeros(1,3, device=device)], dim=-1), h)
    else:
        z = torch.zeros(1, LATENT, device=device)

    # --- action vector (x, y, theta) ---
    act = torch.zeros(1,3, device=device)
    scale_trans = 1.0  # magnitude scaling
    scale_rot   = 1.0

    running = True
    while running:
        # 1. handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # update action from inputs
        if joystick:
            lx = joystick.get_axis(0)  # left stick x (‑1..1)
            ly = -joystick.get_axis(1) # invert y for forward
            rx = joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0
            act = torch.tensor([[lx*scale_trans, ly*scale_trans, rx*scale_rot]], device=device)
        else:
            keys = pygame.key.get_pressed()
            x=y=th=0.0
            if keys[pygame.K_w]: y += 1
            if keys[pygame.K_s]: y -= 1
            if keys[pygame.K_a]: x -= 1
            if keys[pygame.K_d]: x += 1
            if keys[pygame.K_q]: th += 1
            if keys[pygame.K_e]: th -= 1
            act = torch.tensor([[x*scale_trans, y*scale_trans, th*scale_rot]], device=device)

        # 2. rollout one step using PRIOR
        with torch.no_grad():
            # prior from current h
            p_mu, p_lv = rssm.fc_prior(h).chunk(2, dim=-1)
            z = p_mu  # use mean for determinism
            recon = dec(z.unsqueeze(1), h.unsqueeze(1))  # [1,1,3,H,W]
            frame = recon.squeeze(0).squeeze(0).cpu()
            # update h for next step
            h = rssm.rnn(torch.cat([z, act], dim=-1), h)

        # 3. blit frame to window (denormalise)
        img = (frame*0.5 + 0.5).clamp(0,1)  # back to 0‑1
        img_np = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(img_np)))  # (H,W,3) → pygame
        surf = pygame.transform.smoothscale(surf, (size,size))
        screen.blit(surf, (0,0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()

if __name__ == '__main__':
    main()
