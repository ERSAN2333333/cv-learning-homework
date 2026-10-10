import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
w2_path=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(w2_path))
from W3.D06.dataset import trashdataset
from W3.D06.model_resnet50 import ResNet50
from torchvision.models import ResNet50_Weights

backbone_lr, fc_lr = 1e-4, 1e-3
weight_decay, early_stop_patience = 1e-4, 7


def frosen_resnet50(num_classes=6,weights=None):
    weights=ResNet50_Weights.verify(weights)
    model=ResNet50(num_classes=num_classes)
    if weights is not None:
        state_dict=dict(weights.get_state_dict(progress=True,check_hash=True))
        if num_classes==1000:
            model.load_state_dict(state_dict=state_dict,strict=True)
        else:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            result=model.load_state_dict(state_dict,strict=False)
            if set(result.missing_keys)!={'fc.weight','fc.bias'} or result.unexpected_keys:
                raise RuntimeError(f"预训练权重与模型结构不匹配：{result}")
    return model


# 【修改】Compose用圆括号调用，内部传入变换列表。
data_transforms={"train":transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
            "val":transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        }

train_path="/Users/ersan/Downloads/dataset/dataset-split/train"
val_path="/Users/ersan/Downloads/dataset/dataset-split/val"

train_dataset=trashdataset(train_path,transform=data_transforms["train"])
val_dataset=trashdataset(val_path,transform=data_transforms["val"])

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)

num_classes=len(train_dataset.name_list)
weights=ResNet50_Weights.IMAGENET1K_V2

model=frosen_resnet50(num_classes,weights)
loss_fn=torch.nn.CrossEntropyLoss()
# 【修改】startswith是方法，使用圆括号调用。
backbone_parameters=[
    parameters for name,parameters in model.named_parameters() if not name.startswith("fc.")
]
# 【修改】修正paramters拼写为parameters。
optimizer=torch.optim.Adam(model.fc.parameters(),fc_lr)

model.requires_grad_(False)
model.fc.requires_grad_(True)

model.eval()
model.fc.train()

# 【新增】核验梯度开关
for name, parameter in model.named_parameters():
    assert parameter.requires_grad == name.startswith("fc.")

# 【新增】核验所有BN保持eval
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        assert not module.training

# 【修改】同一小批次只更新一次，核验主干不变、分类头有梯度且更新。
if __name__ == "__main__":
    image, label = next(iter(train_loader))
    model.eval()
    model.fc.train()

    # 保存更新前的参数及BN状态。
    before = {name: value.clone() for name, value in model.state_dict().items()}

    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = loss_fn(logits, label)
    loss.backward()

    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            assert parameter.grad is not None, f"{name}没有梯度"
            assert torch.isfinite(parameter.grad).all(), f"{name}梯度异常"
            assert torch.count_nonzero(parameter.grad).item() > 0, f"{name}梯度全为0"
        else:
            assert parameter.grad is None, f"{name}出现了主干梯度"

    optimizer.step()

    for name, value in model.state_dict().items():
        if name.startswith("fc."):
            assert not torch.equal(value, before[name]), f"{name}未更新"
        else:
            assert torch.equal(value, before[name]), f"主干状态{name}发生变化"

    print("核验通过：主干未变，分类头有梯度且已更新一次。")
