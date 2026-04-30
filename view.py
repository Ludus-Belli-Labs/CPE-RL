import torch

obj = torch.load("checkpoints/model_ep420.pt", map_location="cpu")
print(type(obj))
print(obj)