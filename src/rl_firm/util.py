import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape, eps=1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, correction=False)
        batch_count = torch.tensor(x.shape[0], dtype=self.count.dtype, device=x.device)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count) / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-6))
        self.count.copy_(tot_count)

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)