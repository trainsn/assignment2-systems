
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_systems.benchmark import create_model, small_config


def profile_memory(model_size, context_length, mode, precision):
    device = "cuda:0"

    # 1. Model configuration
    config = small_config.copy()

    if model_size == "xl":
        config.update({
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        })

    config["context_length"] = context_length

    # 2. Initialize model, optimizer, and data
    model = create_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

    inputs = torch.randint(
        0, config["vocab_size"],
        (4, context_length),
        device=device,
    )
    targets = torch.randint(
        0, config["vocab_size"],
        (4, context_length),
        device=device,
    )

    # 3. Start memory profiling
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history(max_entries=50000)

    try:
        if mode == "full":
            optimizer.zero_grad(set_to_none=True)

        # 4. Forward
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(precision == "bf16"),
        ):
            logits = model(inputs)

            if mode == "full":
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated() / 1024**3

        # 5. Backward and optimizer
        if mode == "full":
            loss.backward()
            torch.cuda.synchronize()

            after_backward = torch.cuda.memory_allocated() / 1024**3
            peak_through_backward = torch.cuda.max_memory_allocated() / 1024**3

            optimizer.step()
            torch.cuda.synchronize()

            after_optimizer = torch.cuda.memory_allocated() / 1024**3

        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

        print(f"Forward peak: {forward_peak:.2f} GiB")

        if mode == "full":
            print(f"After backward: {after_backward:.2f} GiB")
            print(f"Peak through backward: {peak_through_backward:.2f} GiB")
            print(f"After optimizer: {after_optimizer:.2f} GiB")

        print(f"Overall peak allocated: {peak_allocated:.2f} GiB")
        print(f"Overall peak reserved: {peak_reserved:.2f} GiB")

        # 6. Export memory snapshot
        output_dir = Path("memory_profiles")
        output_dir.mkdir(exist_ok=True)

        filename = output_dir / (
            f"{model_size}_{context_length}_{mode}_{precision}.pickle"
        )

        torch.cuda.memory._dump_snapshot(str(filename))
        print(f"Saved snapshot: {filename}")

    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["small", "xl"], default="small")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "full"], default="forward")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")

    args = parser.parse_args()

    profile_memory(
        args.model,
        args.context_length,
        args.mode,
        args.precision,
    )
