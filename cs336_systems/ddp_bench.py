import os
import time 
import statistics
import argparse

import torch 
import torch.nn.functional as F
import torch.distributed as dist

from cs336_systems.benchmark import create_model, small_config
from cs336_systems.ddp import NaiveDDP

xl_config = {
    **small_config,
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}

def train_step(model, inputs, targets, optimizer):
    torch.cuda.synchronize()
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.nvtx.range("forward"):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
    
    with torch.cuda.nvtx.range("backward"):
        loss.backward()
    
    torch.cuda.synchronize()
    comm_start = time.perf_counter()

    with torch.cuda.nvtx.range("gradient_sync"):
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
    
    comm_end = time.perf_counter()

    with torch.cuda.nvtx.range("optimizer_step"):
        optimizer.step()
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    step_time = (end - start) * 1000
    comm_time = (comm_end - comm_start) * 1000

    return step_time, comm_time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "xl"], default="xl")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = xl_config if args.model == "xl" else small_config
    
    torch.manual_seed(42 + rank)
    model = create_model(config, device=device)
    model = NaiveDDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

    global_bs = 4
    assert global_bs % world_size == 0
    local_bs = global_bs // world_size

    inputs = torch.randint(
        0, config["vocab_size"], 
        (local_bs, config["context_length"]), 
        device=device
    )

    targets = torch.randint(
        0, config["vocab_size"],
        (local_bs, config["context_length"]),
        device=device
    )

    # Warmup
    for _ in range(args.warmup):
        train_step(model, inputs, targets, optimizer)
    
    dist.barrier()
    torch.cuda.synchronize()

    # Benchmark
    times = []

    for _ in range(args.steps):
        step_time, comm_time = train_step(model, inputs, targets, optimizer)
        times.append((step_time, comm_time))

    results = [None] * world_size
    dist.all_gather_object(results, times)

    if rank == 0:
        all_times = [
            t for rank_times in results 
            for t in rank_times
        ]
        
        mean_step = statistics.mean(t[0] for t in all_times)
        mean_comm = statistics.mean(t[1] for t in all_times)

        print(f"Model: {args.model}")
        print(f"World size: {world_size}")
        print(f"Local batch size: {local_bs}")
        print(f"Step time: {mean_step:.2f} ms")
        print(f"Comm time: {mean_comm:.2f} ms")
        print(f"Comm fraction: {mean_comm / mean_step:.2%}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()