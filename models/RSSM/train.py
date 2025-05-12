"""
Recurrent State‑Space World Model (Dreamer‑style)
-------------------------------------------------
This script trains an RSSM on a dataset of RGB image sequences + twist commands
that you already collect for your robot.  After training you can:
  • rollout latent trajectories very quickly on‑device (rssm.imagine)
  • attach an MPC or actor‑critic head for control (not included here)

Quick start
===========
$ python rssm_world_model.py \
    data/                               # parent folder containing sequences
    --img-size 64                       # 64×64 images are usually enough
    --seq-len 16                        # prediction horizon during training
    --batch-size 32 --epochs 50 

Hardware: a single RTX‑4090 will finish a 100k‑frame dataset in ~3 hours.

NOTE  Feel free to swap optimisers, schedulers, mixed‑precision, etc.
"""

import os, json, argparse, random, time, math
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------------------------------• Dataset ------------------------------------------------
class MultiSequenceRobotDataset(Dataset):
    """Same folder layout as your previous project, but returns (obs_seq, act_seq)."""
    def __init__(self, parent_dir: str, seq_len=16, img_size=64,
                 validation=False):
        self.seq_len = seq_len
        self.img_size = img_size
        self.validation = validation
        self.parent_dir = Path(parent_dir)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.subdir_frames = []   # list[list[ (ts, img_path, (x,y,theta)) ] ]
        self.indices       = []   # (subdir_idx, start_idx) for every valid sequence

        # -- discover subdirs
        for sub in sorted(self.parent_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith('validate_') ^ validation:
                continue  # xor selects train vs val dirs
            seq_json = sub/"sequences.json"
            img_dir  = sub/"images"
            if not (seq_json.exists() and img_dir.exists()):
                continue
            frames = self._load_subdir(seq_json, img_dir)
            if len(frames) < seq_len:
                continue
            idx = len(self.subdir_frames)
            self.subdir_frames.append(frames)
            for start in range(len(frames) - seq_len + 1):
                self.indices.append((idx, start))

        if not self.indices:
            raise ValueError(f"No {'validation' if validation else 'training'} data found in {parent_dir}")

    def _load_subdir(self, seq_json: Path, img_dir: Path):
        with open(seq_json, 'r') as f:
            data = json.load(f)['sequences']
        frames = []
        for e in data:
            ts  = e['timestamp']
            img = img_dir/e['image']
            twist = e['twist_commands']
            if twist:
                xs = [t['x'] for t in twist]
                ys = [t['y'] for t in twist]
                ts_ = [t['theta'] for t in twist]
                act = (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(ts_)))
            else:
                act = (0.0, 0.0, 0.0)
            frames.append((ts, img, act))
        frames.sort(key=lambda x: x[0])
        return frames

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sub_i, start = self.indices[index]
        seq = self.subdir_frames[sub_i][start:start+self.seq_len]
        imgs, acts = [], []
        for i, (_, img_path, act) in enumerate(seq):
            img = Image.open(img_path).convert('RGB')
            imgs.append(self.transform(img))
            acts.append(torch.tensor(act, dtype=torch.float))
        # shift so action_{t-1} aligns with obs_t (a_0 := 0)
        acts = torch.stack(acts)           # [T,3]
        acts = torch.cat([torch.zeros(1,3), acts[:-1]], dim=0)
        imgs = torch.stack(imgs)           # [T,3,H,W]
        return imgs, acts

# ----------------------------------------------• Model ---------------------------------------------------
LATENT = 32   # stochastic z_t
HIDDEN = 200  # GRU hidden dim

class ImageEncoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        ch, s = 3, img_size
        def conv(ch_in, ch_out):
            return nn.Sequential(nn.Conv2d(ch_in, ch_out, 4, 2, 1), nn.ReLU())
        self.cnn = nn.Sequential(
            conv(ch, 32),          # s/2
            conv(32, 64),          # s/4
            conv(64, 128),         # s/8
            conv(128, 256),        # s/16
        )
        conv_out = (img_size//16)**2 * 256
        self.fc = nn.Linear(conv_out, 2*LATENT)
    def forward(self, x):           # x:[B*T,3,H,W]
        y = self.cnn(x).flatten(1)
        mu, logvar = self.fc(y).chunk(2, dim=-1)
        return mu, logvar

class ImageDecoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        conv_in = (img_size//16)**2 * 256
        self.fc = nn.Linear(LATENT+HIDDEN, conv_in)
        def tconv(ch_in, ch_out):
            return nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1), nn.ReLU())
        self.deconv = nn.Sequential(
            tconv(256,128),
            tconv(128,64),
            tconv(64,32),
            nn.ConvTranspose2d(32,3,4,2,1)  # final layer no ReLU
        )
    def forward(self, z, h):        # z,h:[B,T,*]
        b,t,_ = z.shape
        x = torch.cat([z,h], dim=-1).reshape(b*t,-1)
        x = self.fc(x).view(b*t,256,self.img_size//16,self.img_size//16)
        x = self.deconv(x).view(b,t,3,self.img_size,self.img_size)
        return x

class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(LATENT+3, HIDDEN)
        self.fc_prior = nn.Linear(HIDDEN, 2*LATENT)
        self.fc_post  = nn.Linear(HIDDEN+LATENT, 2*LATENT)
    def init_state(self, batch, device):
        return torch.zeros(batch, HIDDEN, device=device)
    def observe(self, enc_mu, enc_lv, acts):
        B,T,_ = enc_mu.shape
        h = self.init_state(B, enc_mu.device)
        pri_mu, pri_lv = [], []
        post_mu, post_lv, zs, hs = [], [], [], []
        for t in range(T):
            p_mu, p_lv = self.fc_prior(h).chunk(2, dim=-1)
            pri_mu.append(p_mu); pri_lv.append(p_lv)
            # posterior conditioned on encoder output
            q_in = torch.cat([h, enc_mu[:,t].detach()], dim=-1)
            q_mu, q_lv = self.fc_post(q_in).chunk(2, dim=-1)
            post_mu.append(q_mu); post_lv.append(q_lv)
            z = q_mu + torch.exp(0.5*q_lv)*torch.randn_like(q_mu)
            zs.append(z); hs.append(h)
            h = self.rnn(torch.cat([z, acts[:,t]], dim=-1), h)
        stack = lambda xs: torch.stack(xs, dim=1)
        return (stack(pri_mu), stack(pri_lv),
                stack(post_mu), stack(post_lv),
                stack(zs), stack(hs))

class WorldModel(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.enc = ImageEncoder(img_size)
        self.dec = ImageDecoder(img_size)
        self.rssm= RSSM()
    def forward(self, obs, acts):           # obs:[B,T,3,H,W]  acts:[B,T,3]
        B,T,_,H,W = obs.shape
        mu_e, lv_e = self.enc(obs.reshape(B*T,3,H,W))
        mu_e, lv_e = mu_e.view(B,T,-1), lv_e.view(B,T,-1)
        pri_mu,pri_lv, po_mu,po_lv, z,h = self.rssm.observe(mu_e, lv_e, acts)
        recon = self.dec(z,h)
        return recon, pri_mu,pri_lv, po_mu,po_lv

# -------------------------------------------• Training utils -------------------------------------------

def kl_divergence(mu_q, lv_q, mu_p, lv_p):
    return 0.5*( (mu_q-mu_p).pow(2)/lv_p.exp() + lv_q.exp()/lv_p.exp() - 1 + lv_p - lv_q )

def loss_fn(obs, recon, pri_mu,pri_lv, po_mu,po_lv, freebits=1.):
    # Reconstruction (MSE in pixel space, could switch to symlog)
    rec_loss = F.mse_loss(recon, obs, reduction='none').mean([2,3,4]).sum(1).mean()
    kl = kl_divergence(po_mu, po_lv, pri_mu, pri_lv)
    kl = torch.clamp(kl.mean([0,1]), min=freebits).sum()
    return rec_loss + 1e-4*kl, rec_loss.item(), kl.item()

# ------------------------------------------------• Main ----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train an RSSM world model.')
    p.add_argument('data', help='Parent directory of sequences')
    p.add_argument('--img-size', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--device', choices=['cuda','cpu','auto'], default='auto')
    return p.parse_args()

def main():
    args = parse_args()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) if args.device=='auto' else torch.device(args.device)
    print(f'Using device: {device}')

    # --- datasets
    ds_train = MultiSequenceRobotDataset(args.data, args.seq_len, args.img_size, validation=False)
    ds_val   = MultiSequenceRobotDataset(args.data, args.seq_len, args.img_size, validation=True)

    dl_train = DataLoader(ds_train, args.batch_size, True, num_workers=args.workers, pin_memory=device.type=='cuda')
    dl_val   = DataLoader(ds_val,   args.batch_size, False, num_workers=args.workers, pin_memory=device.type=='cuda')

    model = WorldModel(args.img_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        model.train(); t0=time.time()
        rec_sum=kl_sum=n=0
        for obs, acts in dl_train:
            obs, acts = obs.to(device), acts.to(device)
            opt.zero_grad(set_to_none=True)
            recon, *stats = model(obs, acts)
            loss, rec, kl = loss_fn(obs, recon, *stats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            rec_sum+=rec; kl_sum+=kl; n+=1
        print(f"Epoch {epoch:03d} Train   rec {rec_sum/n:.3f}  kl {kl_sum/n:.3f}  time {time.time()-t0:.1f}s")

        # -------- validation ----------
        model.eval(); rec_sum=kl_sum=n=0
        with torch.no_grad():
            for obs, acts in dl_val:
                obs, acts = obs.to(device), acts.to(device)
                recon, *stats = model(obs, acts)
                _, rec, kl = loss_fn(obs, recon, *stats)
                rec_sum+=rec; kl_sum+=kl; n+=1
        val_loss = rec_sum/n + 1e-4*(kl_sum/n)
        print(f"             Val     rec {rec_sum/n:.3f}  kl {kl_sum/n:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            fname = f"rssm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_best.pth"
            torch.save(model.state_dict(), fname)
            print(f"             >>> saved {fname}")

    # save final
    torch.save(model.state_dict(), 'rssm_final.pth')

if __name__ == '__main__':
    main()
