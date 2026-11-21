import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import statistics

def create_model(config, device="cuda:0"):
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0
    )
    
    model.to(device)
    model.train()

    return model

small_config = {
    "vocab_size": 10000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
}

def step(model, inputs, targets, optimizer, mode):
    if mode not in ("forward", "forward_backward", "full"):
        raise ValueError(f"Unknown mode: {mode}")

    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.nvtx.range("forward"):
        logits = model(inputs)
    
    if mode == "forward":
        return 
    
    with torch.cuda.nvtx.range("loss"):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1),
        )

    with torch.cuda.nvtx.range("backward"):
        loss.backward()

    if mode == "full":
        with torch.cuda.nvtx.range("optimizer"):
            optimizer.step()


def benchmark(model, inputs, targets, optimizer, mode, w=5, n=10):
    def synchronize():
        if inputs.device.type == "cuda":
            torch.cuda.synchronize(inputs.device)
        
    for _ in range(w):
        step(model, inputs, targets, optimizer, mode)
        synchronize()
    
    times = []

    for _ in range(n):
        synchronize()
        start = timeit.default_timer()

        step(model, inputs, targets, optimizer, mode)

        synchronize()
        elapsed = timeit.default_timer() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0

    return mean, std   

if __name__ == "__main__":
    device = "cuda:0"

    inputs = torch.randint(
        0, 
        small_config["vocab_size"], 
        (4, small_config["context_length"]), 
        device=device
    )

    targets = torch.randint(
        0,
        small_config["vocab_size"],
        (4, small_config["context_length"]),
        device=device
    )

    for mode in ("forward", "forward_backward", "full"):
        torch.manual_seed(42)

        model = create_model(small_config, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        mean, std = benchmark(
            model, inputs, targets, optimizer, 
            mode=mode, w=5, n=10
        )

        print(f"Mode: {mode}, Mean time: {mean:.2f} ms, Std dev: {std:.2f} ms")

        del model, optimizer