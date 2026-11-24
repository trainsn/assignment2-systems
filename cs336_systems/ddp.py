import torch 
import torch.nn as nn 
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super(NaiveDDP, self).__init__()
        self.module = module
        
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param.grad is None:
                continue
            
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            param.grad.div_(world_size)


class FlatDDP(NaiveDDP):
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        grads = [param.grad for param in self.module.parameters() if param.grad is not None]
        
        if not grads:
            return 
        
        flat_grads = _flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.div_(world_size)
        for buf, grad in zip(grads, _unflatten_dense_tensors(flat_grads, grads)):
            buf.copy_(grad)
