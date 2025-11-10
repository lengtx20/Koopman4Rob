import torch
import timeit

x = torch.randn(64, 1, 256)


def squeeze():
    x.squeeze(1)
    x.squeeze(1)


print(timeit.timeit(squeeze, number=1000))
