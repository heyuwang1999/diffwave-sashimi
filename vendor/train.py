import os
import time
# import warnings
# warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# from dataset_sc import load_Speech_commands
# from dataset_ljspeech import load_LJSpeech
from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from generate import generate

from models import construct_model

def distributed_train(rank, num_gpus, group_name, cfg):
    # Initialize logger
    if rank == 0 and cfg.wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )

def train(
    rank, num_gpus,
    diffusion_cfg, model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu,
    # n_samples,
    name=None,
    # mel_path=None,
):
    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    n_samples (int):                audio samples to generate and log per checkpoint
    name (str):                     prefix in front of experiment name
    mel_path (str):                 for vocoding, path to mel spectrograms (TODO generate these on the fly)
    """

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, dataset_cfg, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, unconditional=model_cfg.unconditional)
    print('Data loaded')

    # predefine model
    net = construct_model(model_cfg).cuda()
    print_size(net, verbose=False)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            if model_cfg["unconditional"]:
                audio, _, _ = data
                # load audio
                audio = audio.cuda()
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()

            # back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                # print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                # tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                # tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                wandb.log({
                    'train/loss': reduced_loss,
                    'train/log_loss': np.log(reduced_loss),
                }, step=n_iter)

            # save checkpoint
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                # if model_cfg["unconditional"]:
                #     mel_path = None
                #     mel_name = None
                # else:
                #     assert mel_path is not None
                #     mel_name=generate_cfg.mel_name # "LJ001-0001"
                if not model_cfg["unconditional"]: assert generate_cfg.mel_name is not None
                generate_cfg["ckpt_iter"] = n_iter
                samples = generate(
                    rank, # n_iter,
                    diffusion_cfg, model_cfg, dataset_cfg,
                    name=name,
                    **generate_cfg,
                    # n_samples, n_iter, name,
                    # mel_path=mel_path,
                    # mel_name=mel_name,
                )
                samples = [wandb.Audio(sample.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for sample in samples]
                wandb.log(
                    {'inference/audio': samples},
                    step=n_iter,
                    # commit=False,
                )

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            wandb.log({'train/loss_epoch': epoch_loss, 'train/log_loss_epoch': np.log(epoch_loss)}, step=n_iter)

    # Close logger
    if rank == 0:
        # tb.close()
        wandb.finish()

def _pred_to_x0(prediction, x_t, sqrt_abar, sqrt_1m_abar, parameterization):
    """Recover the clean-signal estimate x0 from the model output."""
    if parameterization == "eps":
        return (x_t - sqrt_1m_abar * prediction) / sqrt_abar
    return sqrt_abar * x_t - sqrt_1m_abar * prediction  # "v"


def multi_resolution_stft_loss(x, y, fft_sizes=(512, 1024, 2048),
                               hop_sizes=(128, 256, 512), win_sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss (Yamamoto et al. 2020): spectral-convergence +
    log-magnitude L1, summed over FFT resolutions. Encourages sharp, clean
    high-frequency detail. x, y: (B, 1, L) or (B, L)."""
    if x.dim() == 3:
        x, y = x.squeeze(1), y.squeeze(1)
    total = 0.0
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win, device=x.device)
        kw = dict(n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                  return_complex=True, center=True, pad_mode="constant")  # constant pad: robust to short clips
        sx, sy = torch.stft(x, **kw).abs(), torch.stft(y, **kw).abs()
        sc = torch.linalg.norm(sy - sx) / (torch.linalg.norm(sy) + 1e-7)
        mag = F.l1_loss(torch.log(sx + 1e-7), torch.log(sy + 1e-7))
        total = total + sc + mag
    return total / len(fft_sizes)


def training_loss(net, loss_fn, audio, diffusion_hyperparams, mel_spec=None):
    """
    Compute the diffusion training loss.

    Supports (all config-gated, defaults == original DiffWave):
      - parameterization: "eps" or "v" (Salimans & Ho 2022)
      - min_snr_gamma: Min-SNR-gamma loss weighting (Hang et al. 2023)
      - stft_loss_weight: multi-resolution STFT auxiliary loss on x0
      - self-conditioning (read from the model): two-pass with a detached x0 estimate

    Parameters:
    net (torch network):            the model
    loss_fn (torch loss function):  default nn.MSELoss()
    audio (torch.tensor):           training data, shape=(batchsize, 1, length)
    diffusion_hyperparams (dict):   from calc_diffusion_hyperparams (cuda tensors)
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    parameterization = _dh.get("parameterization", "eps")
    min_snr_gamma = _dh.get("min_snr_gamma", None)
    stft_w = _dh.get("stft_loss_weight", 0.0)
    self_cond = getattr(net, "self_conditioning", False) or \
        getattr(getattr(net, "module", None), "self_conditioning", False)

    B, C, L = audio.shape
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # sample steps 1~T
    z = torch.normal(0, 1, size=audio.shape).cuda()
    abar = Alpha_bar[diffusion_steps]                       # (B,1,1)
    sqrt_abar, sqrt_1m_abar = torch.sqrt(abar), torch.sqrt(1 - abar)
    x_t = sqrt_abar * audio + sqrt_1m_abar * z              # x_t from q(x_t|x_0)
    steps = diffusion_steps.view(B, 1)

    # Self-conditioning: with prob 0.5, feed a detached x0 estimate from a no-grad
    # pass; otherwise feed zeros (matching the first inference step).
    x_self = None
    if self_cond:
        x_self = torch.zeros_like(audio)
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                pred0 = net((x_t, steps), mel_spec=mel_spec, x_self_cond=x_self)
                x_self = _pred_to_x0(pred0, x_t, sqrt_abar, sqrt_1m_abar, parameterization).detach()

    # Only pass x_self_cond when self-conditioning (WaveNet.forward has no such arg).
    extra = {"x_self_cond": x_self} if self_cond else {}
    prediction = net((x_t, steps), mel_spec=mel_spec, **extra)

    # Fast path: original DiffWave (eps, unweighted, no STFT) — unchanged numerics.
    if parameterization == "eps" and min_snr_gamma is None and stft_w == 0.0:
        return loss_fn(prediction, z)

    if parameterization == "eps":
        target = z
    elif parameterization == "v":
        target = sqrt_abar * z - sqrt_1m_abar * audio  # v = sqrt(abar)*eps - sqrt(1-abar)*x0
    else:
        raise ValueError(f"Unknown parameterization {parameterization!r} (use 'eps' or 'v')")

    se = (prediction - target) ** 2                        # (B,1,L)
    if min_snr_gamma is None:
        loss = se.mean()
    else:
        # Min-SNR-gamma weighting (Hang et al. 2023). SNR = abar / (1-abar).
        snr = abar / (1 - abar)
        clamped = torch.clamp(snr, max=min_snr_gamma)
        weight = clamped / snr if parameterization == "eps" else clamped / (snr + 1)
        loss = (weight * se).mean()

    if stft_w > 0.0:
        x0_hat = _pred_to_x0(prediction, x_t, sqrt_abar, sqrt_1m_abar, parameterization)
        loss = loss + stft_w * multi_resolution_stft_loss(x0_hat, audio)
    return loss



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    os.makedirs("exp/", mode=0o775, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
