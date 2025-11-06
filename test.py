# Source - https://stackoverflow.com/questions/52076815/pytorch-use-device-inside-with-statement
# Posted by Mahnerak
# Retrieved 2025/11/5, License - CC-BY-SA 4.0
import torch


with torch.device("cuda"):
    mod = torch.nn.Linear(20, 30)
    print(mod.weight.device)  # Output: cuda:0
    print(mod(torch.randn(128, 20)).device)  # Output: cuda:0
