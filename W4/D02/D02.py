import torch
import sys
from pathlib import Path
w2_path=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(w2_path))
from W3.D06.dataset import trashdataset
from torchvision.models import ResNet50_Weights,resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
# 【新增：第三题】保存图片和自动记录终端输出，无需额外运行参数。
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from contextlib import redirect_stdout, redirect_stderr
import traceback

train_path="/Users/ersan/Downloads/dataset/dataset-split/train"
val_path="/Users/ersan/Downloads/dataset/dataset-split/val"


weights=ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()  # 获取该权重配套的预处理，主要用于验证推理
train_transform=transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class Tee:
    """【新增：第三题】同一份实际输出同时显示在终端并写入日志。"""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def main(out_dir):
    # 【修改：第三题】原主流程放入main，确保直接运行时输出可自动记录。
    train_dataset=trashdataset(train_path,transform=train_transform)
    val_dataset=trashdataset(val_path,transform=preprocess)
    print("W04-D02-Q03：真实运行日志")
    print("权重来源：torchvision官方ImageNet预训练权重")
    print("权重枚举：", weights)
    print(len(train_dataset))
    print(len(val_dataset))
    train_class_to_idx={name:idx for idx,name in enumerate(train_dataset.name_list)}
    val_class_to_idx={name:idx for idx,name in enumerate(val_dataset.name_list)}
    print("训练集类别映射：", train_class_to_idx)
    print("验证集类别映射：", val_class_to_idx)
    assert train_class_to_idx == val_class_to_idx, "训练和验证类别映射不一致"
    print("检查通过：训练和验证类别映射完全一致")   


    
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)
    
    #测试一张图片经过预处理的
    image,label=val_dataset[0]
    x=image.unsqueeze(0)
    print("训练预处理操作:",train_transform)
    print("验证推理预处理操作:",preprocess)
    print("输入形状：", x.shape)   # [1, 3, 224, 224]
    print("数值范围：", x.min().item(), x.max().item())

    # 【新增：第三题】保存与val_dataset[0]同源的RGB原图。
    source_path = val_dataset.image_path[0]
    before_path = out_dir / "Q03_before.png"
    after_path = out_dir / "Q03_after.png"
    with Image.open(source_path) as original:
        original.convert("RGB").save(before_path)

    # 【新增：第三题】反归一化仅用于展示；不修改image/x或模型输入。
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    display_image = (image * std + mean).clamp(0, 1)
    to_pil_image(display_image).save(after_path)
    print("同图来源：", source_path)
    print("变换前图片：", before_path)
    print("变换后展示图：", after_path)

    # 【新增：第三题】保存训练集、验证集的类别编号对照表。
    rows = ["类别\ttrain编号\tval编号"]
    for name in train_dataset.name_list:
        rows.append(f"{name}\t{train_class_to_idx[name]}\t{val_class_to_idx[name]}")
    table_path = out_dir / "Q03_classes.txt"
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("类别对照表：", table_path)

    # 【修改：说明】替换分类头，保留主干预训练权重；新头随机初始化。
    model=resnet50(weights=weights)
    num_classes=len(train_dataset.name_list)
    print("分类头输入维度：", model.fc.in_features)
    model.fc=torch.nn.Linear(model.fc.in_features,num_classes)
    images,labels=next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        logits=model(images)
    
    print("图片形状：", images.shape)
    print("logits形状：", logits.shape)
    print("标签形状：", labels.shape)
    print("标签类型：", labels.dtype)
    print("标签范围：", labels.min().item(), labels.max().item())
    # 检查输入、输出和标签接口
    assert images.ndim == 4 and images.shape[1] == 3, "输入不是RGB批次"
    assert logits.shape == (images.shape[0], num_classes), "输出形状错误"
    assert labels.shape == (images.shape[0],), "标签形状错误"
    assert labels.dtype == torch.long, "标签类型错误"
    assert labels.min().item() >= 0 and labels.max().item() < num_classes, "标签越界"

    print("单批前向检查通过")


if __name__=="__main__":
    # 【新增：第三题】输出均保存到当前脚本目录；再次运行会刷新这些材料。
    out_dir = Path(__file__).resolve().parent
    log_path = out_dir / "Q03_forward_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            try:
                main(out_dir)
                print("实际运行日志已保存：", log_path)
            except Exception:
                # 【新增：第三题】失败时也保存真实错误，不误写“检查通过”。
                traceback.print_exc()
                raise SystemExit(1)
