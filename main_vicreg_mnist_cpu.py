# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor 

import augmentations as aug
from distributed import init_distributed_mode


import resnet


def get_arguments():
  parser = argparse.ArgumentParser(
      description="Pretrain a resnet model with VICReg", add_help=False)

  # Data
  parser.add_argument("--data_dir",
                      type=Path,
                      default="/hdd/zengs_data/vision_data",
                      required=True,
                      help='Path to the image net dataset')

  # Checkpoints
  parser.add_argument(
      "--exp_dir",
      type=Path,
      default="./exp_mnist",
      help=
      'Path to the experiment folder, where all logs/checkpoints will be stored'
  )
  parser.add_argument(
      "--log_freq_time",
      type=int,
      default=60,
      help='Print logs to the stats.txt file every [log-freq-time] seconds')

  # Model
  parser.add_argument("--arch",
                      type=str,
                      default="resnet50",
                      help='Architecture of the backbone encoder network')
  parser.add_argument("--mlp",
                      default="8192-8192-8192",
                      help='Size and number of layers of the MLP expander head')

  # Optim
  parser.add_argument("--epochs",
                      type=int,
                      default=100,
                      help='Number of epochs')
  parser.add_argument(
      "--batch_size",
      type=int,
      # default=2048,
      default=128,
      help=
      'Effective batch size (per worker batch size is [batch-size] / world-size)'
  )
  parser.add_argument(
      "--base_lr",
      type=float,
      default=0.2,
      help=
      'Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256'
  )
  parser.add_argument("--wd", type=float, default=1e-6, help='Weight decay')

  # Loss
  parser.add_argument("--sim_coeff",
                      type=float,
                      default=25.0,
                      help='Invariance regularization loss coefficient')
  parser.add_argument("--std_coeff",
                      type=float,
                      default=25.0,
                      help='Variance regularization loss coefficient')
  parser.add_argument("--cov_coeff",
                      type=float,
                      default=1.0,
                      help='Covariance regularization loss coefficient')

  # Running
  parser.add_argument("--num_workers", type=int, default=2)
  parser.add_argument('--device',
                      default='cuda',
                      help='device to use for training / testing')

  # Distributed
  parser.add_argument('--world_size',
                      default=10,
                      type=int,
                      help='number of distributed processes')
  parser.add_argument('--local_rank', default=-1, type=int)
  parser.add_argument('--dist_url',
                      default='env://',
                      help='url used to set up distributed training')

  return parser

# python main_vicreg_mnist_cpu.py --exp_dir cpu_mnist --device cpu --data_dir ./data 

def main(args):
  
  gpu = torch.device(args.device)
  
  args.exp_dir.mkdir(parents=True, exist_ok=True)
  stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
  print(" ".join(sys.argv))
  print(" ".join(sys.argv), file=stats_file)

  # transforms = aug.TrainTransform()
  transforms = aug.TrainTransformMNIST()

  # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
  dataset = datasets.MNIST(args.data_dir,
                           download=True,
                           train=True,
                           transform = transforms)
  # sampler = torch.utils.data.distributed.DistributedSampler(dataset,
  #                                                           shuffle=True)
  sampler = None

  per_device_batch_size = args.batch_size
  loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=per_device_batch_size,
      num_workers=args.num_workers,
      pin_memory=True,
      sampler=sampler,
  )
  
  model = VICReg(args)
  # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
  optimizer = LARS(
      model.parameters(),
      lr=0,
      weight_decay=args.wd,
      weight_decay_filter=exclude_bias_and_norm,
      lars_adaptation_filter=exclude_bias_and_norm,
  )
  

  if (args.exp_dir / "model.pth").is_file():
    if args.rank == 0:
      print("resuming from checkpoint")
    ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    start_epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
  else:
    start_epoch = 0

  start_time = last_logging = time.time()
  scaler = torch.cuda.amp.GradScaler()
  for epoch in range(start_epoch, args.epochs):

    # sampler.set_epoch(epoch)
    
    for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
      
      lr = adjust_learning_rate(args, optimizer, loader, step)

      optimizer.zero_grad()
      loss = model.forward(x, y)
      loss.backward()
      optimizer.step()

      current_time = time.time()
      if current_time - last_logging > args.log_freq_time:
        stats = dict(
            epoch=epoch,
            step=step,
            loss=loss.item(),
            time=int(current_time - start_time),
            lr=lr,
        )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)
        last_logging = current_time

    state = dict(
        epoch=epoch + 1,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    torch.save(state, args.exp_dir / "model.pth")

  torch.save(model.module.backbone.state_dict(),
             args.exp_dir / "resnet50.pth")


def adjust_learning_rate(args, optimizer, loader, step):
  max_steps = args.epochs * len(loader)
  warmup_steps = 10 * len(loader)
  base_lr = args.base_lr * args.batch_size / 256
  if step < warmup_steps:
    lr = base_lr * step / warmup_steps
  else:
    step -= warmup_steps
    max_steps -= warmup_steps
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    end_lr = base_lr * 0.001
    lr = base_lr * q + end_lr * (1 - q)
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
  return lr


class VICReg(nn.Module):

  def __init__(self, args):
    super().__init__()
    self.args = args
    self.num_features = int(args.mlp.split("-")[-1])
    self.backbone, self.embedding = resnet.__dict__[args.arch](num_channels=1,
        zero_init_residual=True)
    self.projector = Projector(args, self.embedding)

  def forward(self, x, y):
    x = self.projector(self.backbone(x))
    y = self.projector(self.backbone(y))

    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(
        F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (self.args.batch_size - 1)
    cov_y = (y.T @ y) / (self.args.batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
        self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(
            self.num_features)

    loss = (self.args.sim_coeff * repr_loss + self.args.std_coeff * std_loss +
            self.args.cov_coeff * cov_loss)
    return loss


def Projector(args, embedding):
  mlp_spec = f"{embedding}-{args.mlp}"
  layers = []
  f = list(map(int, mlp_spec.split("-")))
  for i in range(len(f) - 2):
    layers.append(nn.Linear(f[i], f[i + 1]))
    layers.append(nn.BatchNorm1d(f[i + 1]))
    layers.append(nn.ReLU(True))
  layers.append(nn.Linear(f[-2], f[-1], bias=False))
  return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
  return p.ndim == 1


def off_diagonal(x):
  n, m = x.shape
  assert n == m
  return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):

  def __init__(
      self,
      params,
      lr,
      weight_decay=0,
      momentum=0.9,
      eta=0.001,
      weight_decay_filter=None,
      lars_adaptation_filter=None,
  ):
    defaults = dict(
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        eta=eta,
        weight_decay_filter=weight_decay_filter,
        lars_adaptation_filter=lars_adaptation_filter,
    )
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self):
    for g in self.param_groups:
      for p in g["params"]:
        dp = p.grad

        if dp is None:
          continue

        if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
          dp = dp.add(p, alpha=g["weight_decay"])

        if g["lars_adaptation_filter"] is None or not g[
            "lars_adaptation_filter"](p):
          param_norm = torch.norm(p)
          update_norm = torch.norm(dp)
          one = torch.ones_like(param_norm)
          q = torch.where(
              param_norm > 0.0,
              torch.where(update_norm > 0,
                          (g["eta"] * param_norm / update_norm), one),
              one,
          )
          dp = dp.mul(q)

        param_state = self.state[p]
        if "mu" not in param_state:
          param_state["mu"] = torch.zeros_like(p)
        mu = param_state["mu"]
        mu.mul_(g["momentum"]).add_(dp)

        p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
  x_list = FullGatherLayer.apply(x)
  return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
  """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

  @staticmethod
  def forward(ctx, x):
    output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x)
    return tuple(output)

  @staticmethod
  def backward(ctx, *grads):
    all_gradients = torch.stack(grads)
    dist.all_reduce(all_gradients)
    return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
  os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
  exit()


def handle_sigterm(signum, frame):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser('VICReg training script',
                                   parents=[get_arguments()])
  args = parser.parse_args()
  main(args)
