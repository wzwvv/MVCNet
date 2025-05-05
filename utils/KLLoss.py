# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/20 18:28
@Auth ： S Yang
@File ：KLLoss.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn


# confirm pre and pre_hat is softmax tensors
def jug(pre: torch.Tensor):
    if pre.dim() != 2:
        return False
    if (pre < 0).any():
        return False

    # 检查和是否为1
    if not torch.allclose(pre.sum(dim=0), torch.tensor(1.0)):
        return False
    return True


class KLLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pre_hat, pre):
        assert jug(pre_hat) and jug(pre), 'input must be a softmax tensor'
        return (pre_hat * torch.log((pre_hat + self.epsilon) / (pre + self.epsilon))).sum(dim=1).sum() / pre_hat.size(0)

