import os 
import time
import statistics

import torch 
import torch.distributed as dist 

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for size_mb in (1, 10, 100, 1024):
        num_elems = size_mb * 1024 * 1024 // 4
        x = torch.ones(
            num_elems, 
            device=f"cuda:{local_rank}",
            dtype=torch.float32
        )

        # Warmup
        for _ in range(5):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        dist.barrier()

        times = []

        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()

            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)
        
        results = [None] * world_size
        dist.all_gather_object(results, times)

        if rank == 0:
            means = [statistics.mean(r) for r in results]
            print(f"Size: {size_mb} MB, Mean time: {means} ms")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()