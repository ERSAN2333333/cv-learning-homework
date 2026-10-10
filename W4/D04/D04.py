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
import subprocess

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

model.requires_grad_(False)
model.fc.requires_grad_(True)
loss_fn=torch.nn.CrossEntropyLoss()
# 【修改】startswith是方法，使用圆括号调用。
backbone_parameters=[
    parameters for name,parameters in model.named_parameters() if not name.startswith("fc.")
]
# 【修改】修正paramters拼写为parameters。
optimizer=torch.optim.Adam(model.fc.parameters(),fc_lr)


save_dir=Path(__file__).resolve().parent

config=(
    f"解释器:{sys.executable}\n"
    f"Python版本:{sys.version}\n"
    f"设备:{next(model.parameters()).device}\n"
    f"预训练权重：{weights.name}\n"
    f"训练数据：{train_path}\n"
    f"类别表：{train_dataset.name_list}\n"
    f"batch_size：{train_loader.batch_size}\n"
    f"优化器：{optimizer}\n"
    f"训练范围：仅分类头，主干及BN保持eval\n"
    f"训练变换：{data_transforms['train']}\n"
    f"测试规模：一个批次，更新一次\n"
)
(save_dir/"config.txt").write_text(config,encoding='utf-8')

dependencies=subprocess.check_output(
    [sys.executable,"-m","pip","freeze"],
    text=True
)
(save_dir/"requirements.txt").write_text(  
    dependencies,encoding='utf-8'
)

if __name__ == "__main__":
    image, label = next(iter(train_loader))
    model.eval()
    model.fc.train()

    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = loss_fn(logits, label)
    loss.backward()
    optimizer.step()

    
    # 【修改】保存一次小批次测试的真实日志；不是完整的一个epoch。
    log_text = (
        f"小测试1/1：step=1，样本数={label.size(0)}，"
        f"更新前loss={loss.item():.6f}，一次更新已完成\n"
    )
    print(log_text, end="")
    (save_dir / "train_log.txt").write_text(log_text, encoding="utf-8")

    # 【新增】更新后切换到eval，记录同一固定输入的基准输出。
    model.eval()
    with torch.no_grad():
        reference_logits = model(image)

    # 【新增】优化器没有.item()；用state_dict()保存它的状态。
    checkpoint = {
        "step": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes": train_dataset.name_list,
        "config": config,
        "fixed_images": image,
        "fixed_labels": label,
        "reference_logits": reference_logits,
    }
    # 本次只有一个检查点，best和last相同；没有进行验证集择优。
    torch.save(checkpoint, save_dir / "last.pth")
    torch.save(checkpoint, save_dir / "best.pth")
    fields = "权重字段：" + ", ".join(checkpoint.keys()) + "\n"
    note = "本次仅一次小批次更新，best=last，未进行验证集择优。\n"
    with (save_dir / "train_log.txt").open("a", encoding="utf-8") as file:
        file.write(fields + note)
    print(fields + note, end="")

    # 【新增】使用同一个解释器启动新进程，不需要手动添加命令行参数。
    subprocess.run([sys.executable, str(save_dir / "reload_test.py")], check=True)
