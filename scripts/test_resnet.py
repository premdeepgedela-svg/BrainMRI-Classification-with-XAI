import torch

from src.models.resnet18 import BrainMRIResNet18

model = BrainMRIResNet18()

x = torch.randn(8, 4, 240, 240)

y = model(x)

print("=" * 60)
print(model)
print("=" * 60)
print("Input :", x.shape)
print("Output:", y.shape)