import torch
from torch.utils.data.distributed import DistributedSampler
def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    # TODO would be nice if unconditional was decoupled from dataset
    # NOTE: dataset modules are imported lazily inside each branch. sc.py pulls in
    # torchvision and a torchaudio download API removed in torchaudio>=2.x, so a
    # top-level import would break the `music` path on modern stacks (incl. Colab).

    dataset_name = dataset_cfg.pop("_name_")
    if dataset_name == "sc09":
        assert unconditional
        from .sc import SpeechCommands
        dataset = SpeechCommands(dataset_cfg.data_path)
    elif dataset_name == "ljspeech":
        assert not unconditional
        from .mel2samp import Mel2Samp
        dataset = Mel2Samp(**dataset_cfg)
    elif dataset_name == "music":
        # Project addition: unconditional raw-waveform music from a few long files.
        assert unconditional, "music dataset currently supports unconditional generation (milestone 1)"
        from .music import MusicWaveform
        dataset = MusicWaveform(**dataset_cfg)
    elif dataset_name == "music_continuation":
        # Project addition (milestone 2): (target, context) pairs for continuation.
        from .music import MusicContinuation
        dataset = MusicContinuation(**dataset_cfg)
    dataset_cfg["_name_"] = dataset_name # Restore

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    return trainloader
