import torch

obj = torch.load("checkpoints/model_ep400.json", map_location="cpu")
print(type(obj))
print(obj)