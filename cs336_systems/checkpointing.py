
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cs336_systems.benchmark import create_model, small_config


class CheckpointGroup(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        def run_group(y):
            for block in self.blocks:
                y = block(y)
            return y

        return checkpoint(
            run_group, x, use_reentrant=False
        )


def apply_checkpointing(model, group_size):
    if group_size == 0:
        return model  # No checkpointing


    blocks = list(model.layers)

    groups = [
        CheckpointGroup(blocks[i:i + group_size])
        for i in range(0, len(blocks), group_size)
    ]

    model.layers = nn.ModuleList(groups)
    return model


def run_experiment(model_size, seq_len, group_size):
    device = "cuda:0"

    config = small_config.copy()

    if model_size == "xl":
        config.update({
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        })

    config["context_length"] = seq_len

    torch.manual_seed(42)
    model = create_model(config, device)

    model = apply_checkpointing(model, group_size)

    inputs = torch.randint(
        0, config["vocab_size"],
        (4, seq_len),
        device=device,
    )

    targets = torch.randint(
        0, config["vocab_size"],
        (4, seq_len),
        device=device,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    baseline = torch.cuda.memory_allocated(device)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    forward_peak = torch.cuda.max_memory_allocated(device)

    loss.backward()
    torch.cuda.synchronize()

    training_peak = torch.cuda.max_memory_allocated(device)

    gib = 1024 ** 3

    print(f"Group size: {group_size}")
    print(f"Baseline memory: {baseline/gib:.2f} GiB")
    print(f"Forward peak: {forward_peak/gib:.2f} GiB")
    print(f"Forward+Backward peak: {training_peak/gib:.2f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", choices=["small", "xl"], default="small"
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=2)

    args = parser.parse_args()

    run_experiment(
        args.model,
        args.seq_len,
        args.group_size,
    )
