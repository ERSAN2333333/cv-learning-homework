import torch
from model_resnet50 import resnet50

model = resnet50(num_classes=6)  # 不需要预训练权重
model.eval()

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    model(x)

# 1. 统计每一层自身的参数量
for name, layer in model.named_modules():
    # recurse=False：不重复统计子层的参数
    count = sum(p.numel() for p in layer.parameters(recurse=False))
    if count > 0:
        print(f"{name}：{count} 个参数")

# 2. 统计整个模型的参数量
total = sum(p.numel() for p in model.parameters())

# 3. 统计允许训练的参数量
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"总参数量：{total}")
print(f"可训练参数量：{trainable}")