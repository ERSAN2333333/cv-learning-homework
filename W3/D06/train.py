"""ResNet50 六分类微调；先运行 split_dataset.py 完成数据划分。"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from dataset import CLASS_NAMES, trashdataset
from model_resnet50 import resnet50


# 【修改】把训练、验证、测试共用的计算放到一个函数，减少重复代码。
# optimizer 不为空时训练；为空时评估，关闭梯度并切换到 eval 模式。
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_num = 0.0, 0
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            # 【修改】模型、图片和标签必须位于同一个设备。
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # 【保留】模型输出 logits，直接交给 CrossEntropyLoss，不提前 softmax。
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            size = labels.size(0)
            # 【修改】沿用你已改好的 .item()，只累计数值，避免保留计算图引用。
            total_loss += loss.item() * size
            total_num += size
            # 【新增】混淆矩阵：行是真实类别，列是预测类别；指标统计放在 CPU。
            actual = labels.detach().cpu()
            predicted = logits.detach().argmax(dim=1).cpu()
            indices = actual * len(CLASS_NAMES) + predicted
            confusion += torch.bincount(
                indices, minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    # 【修改】整轮结束后统一计算，避免在批次循环中反复计算 acc。
    if total_num == 0:
        raise ValueError("DataLoader 没有样本")
    matrix = confusion.float()
    true_positive = matrix.diag()
    # 【新增】各类召回率 = 预测正确数 / 该类真实样本数。
    recall = true_positive / matrix.sum(dim=1).clamp_min(1)
    # 【新增】F1 = 2TP / (2TP + FP + FN)；macro-F1 对六个类别等权平均。
    # 类别数量不均衡时，同时看 macro-F1，避免只看整体准确率。
    f1 = 2 * true_positive / (matrix.sum(dim=0) + matrix.sum(dim=1)).clamp_min(1)
    return {
        "loss": total_loss / total_num,
        "acc": true_positive.sum().item() / total_num,
        "macro_f1": f1.mean().item(),
        "class_recall": recall.tolist(),
    }


def main():
    # 【修改】把配置集中在 main 中；以下是小数据集全模型微调的起点。
    # 骨干已有预训练参数，lr 较小；fc 从随机参数开始，lr 较大。
    # batch_size 从 32 调到 16，降低内存需求；最多训练 30 轮。
    seed = 42
    batch_size = 16
    epochs = 30
    backbone_lr, fc_lr = 1e-4, 1e-3
    weight_decay, early_stop_patience = 1e-4, 7
    # 【修改】指向划分后的目录；train / val / test 由这个根目录派生。
    data_root = Path("/Users/ersan/Downloads/dataset/dataset-split")
    # 【修改】保存路径以本脚本为基准，避免从不同工作目录运行时保存到别处。
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    # 【新增】固定随机种子，便于比较实验；不保证所有硬件上逐位一致。
    random.seed(seed)
    torch.manual_seed(seed)
    # 【新增】优先 CUDA，其次 Mac 的 MPS，否则使用 CPU。
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("设备：", device)

    weights = ResNet50_Weights.IMAGENET1K_V2
    train_transform = transforms.Compose([
        # 【修改】默认最小裁剪面积 0.08 改为 0.7，降低把垃圾物体裁掉的概率。
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # 【保留】验证和测试沿用 V2 官方预处理，避免重复 ToTensor / Normalize。
    val_transform = weights.transforms()
    # 【新增】先检查划分清单，再加载模型，路径错误时不会先下载权重。
    manifest_path = data_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("未找到划分清单，请先运行 split_dataset.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["classes"] != CLASS_NAMES:
        raise ValueError("划分清单和 Dataset 的类别顺序不同")
    # 【修改】定义原来缺失的三个数据目录。
    train_root, val_root, test_root = (data_root / name for name in ("train", "val", "test"))
    train_dataset = trashdataset(train_root, train_transform)
    val_dataset = trashdataset(val_root, val_transform)
    test_dataset = trashdataset(test_root, val_transform)
    # 【修改】DataLoader 必须传 dataset，不能传 lr；lr 属于优化器。
    # 训练集 shuffle=True；验证和测试 shuffle=False；先用 num_workers=0。
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    # 【修改】检查数据路径后再创建模型，并把模型移到选定设备。
    # 加载函数已经跳过官方的 1000 类 fc，此处自动创建 6 类 fc。
    model = resnet50(num_classes=len(CLASS_NAMES), weights=weights).to(device)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ]
    # 【修改】Adam 改为 AdamW，增加 weight_decay，并分别设置骨干 / fc 学习率。
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": backbone_lr},
        {"params": model.fc.parameters(), "lr": fc_lr},
    ], weight_decay=weight_decay)
    # 【新增】验证 macro-F1 停滞时降低学习率；mode="max" 表示指标越大越好。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": seed, "batch_size": batch_size, "epochs": epochs,
        "backbone_lr": backbone_lr, "fc_lr": fc_lr,
        "weight_decay": weight_decay, "crop_scale": [0.7, 1.0],
        "early_stop_patience": early_stop_patience, "num_workers": 0,
        "weights": "IMAGENET1K_V2",
        "scheduler": {"type": "ReduceLROnPlateau", "mode": "max",
                      "factor": 0.5, "patience": 2},
    }
    best_f1, no_improvement = -1.0, 0
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["macro_f1"])
        improved = val_metrics["macro_f1"] > best_f1
        if improved:
            best_f1, no_improvement = val_metrics["macro_f1"], 0
        else:
            no_improvement += 1
        # 【修改】用 f-string 输出真实数值，原 print 中的 {} 不会自动替换。
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}, "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )
        # 【修改】保存参数、优化器、调度器、类别顺序、划分清单和配置。
        checkpoint = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": config, "classes": CLASS_NAMES.copy(), "split_manifest": manifest,
            "train": train_metrics, "val": val_metrics, "best_macro_f1": best_f1,
            "no_improvement": no_improvement,
        }
        # 【修改】nn.Module 没有 model.save()，使用 torch.save()。
        # 只保留最佳和最新两个 checkpoint，避免每轮都增加大文件。
        # 在本轮立即写入，避免 state_dict 引用随后被训练更新。
        if improved:
            torch.save(checkpoint, checkpoint_dir / "best_resnet50.pth")
        torch.save(checkpoint, checkpoint_dir / "last_resnet50.pth")
        # 【新增】连续 7 轮没有刷新验证 macro-F1 时提前停止。
        if no_improvement >= early_stop_patience:
            print(f"验证 macro-F1 连续 {early_stop_patience} 轮未提高，提前结束")
            break

    # 【新增】测试集不参与调参或模型选择；最后加载最佳模型评估一次。
    best = torch.load(
        checkpoint_dir / "best_resnet50.pth", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print(f"测试集：loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}, "
          f"macro-F1={test_metrics['macro_f1']:.4f}")
    for name, recall in zip(CLASS_NAMES, test_metrics["class_recall"]):
        print(f"{name} 召回率：{recall:.4f}")


# 【修改】只有直接运行此文件才启动训练；被 import 时不会下载权重或训练。
if __name__ == "__main__":
    main()
