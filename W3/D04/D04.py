import sys
from pathlib import Path
import torch

# 导入你在D03.py中定义的模型，不会执行其中的训练循环。
sys.path.insert(0, "/Users/ersan/rs_cvpy/cv_learning/W3/D03")
from D03 import resnet18, FlowerData
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
import json

def run_experiment(model, optimizer, train_dataset, val_dataset,
                   device, epochs, group_name, save_dir):

    # 每组重新创建相同种子的加载器，保证训练样本打乱顺序一致。
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    lr = optimizer.param_groups[0]["lr"]

    update_steps = 0
    samples_seen = 0
    best_acc = -1
    best_epoch = 0
    history = []

    # MPS计算是异步的，计时前等待之前的计算完成。
    torch.mps.synchronize()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        total_samples = 0

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            # 每执行一次step，就更新一次参数。
            update_steps += 1

            # 累计训练中实际处理的样本次数，包含重复轮次。
            samples_seen += len(label)

            total_loss += loss.item() * len(label)
            total_samples += len(label)

        train_loss = total_loss / total_samples

        # ---------- 验证：不反传、不更新参数 ----------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                pred = model(image)
                predicted_label = pred.argmax(dim=1)

                correct += (predicted_label == label).sum().item()
                total += len(label)

        val_acc = correct / total

        print(
            f"{group_name} 第{epoch + 1}轮，"
            f"训练loss：{train_loss:.4f}，"
            f"验证准确率：{val_acc:.2%}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc
        })

        # 两组都只按验证准确率选best，规则完全相同。
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "val_acc": best_acc,
                "lr": lr
            }

            # A组和B组使用不同文件名。
            torch.save(
                checkpoint,
                save_dir / f"{group_name}_best.pth"
            )

    # 等待MPS计算完成，再停止计时。
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time

    record = {
        "group": group_name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": 32,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "update_steps": update_steps,
        "samples_seen": samples_seen,
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "history": history
    }

    with open(
        save_dir / f"{group_name}_record.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(record, file, ensure_ascii=False, indent=4)

    print(f"{group_name} 更新步数：{update_steps}")
    print(f"{group_name} 累计训练样本次数：{samples_seen}")
    print(f"{group_name} 耗时：{elapsed:.2f}秒")
    print(f"{group_name} 最佳轮次：{best_epoch}")
    print(f"{group_name} 最佳验证准确率：{best_acc:.2%}")


if __name__ == "__main__":
    device = torch.device("mps")

    # 一、创建未训练的模型，保存初始状态。
    torch.manual_seed(42)
    initial_model = resnet18(num_classes=5)

    # 保存到当前脚本所在文件夹。
    save_path = Path(__file__).resolve().parent / "initial.pth"
    torch.save(initial_model.state_dict(), save_path)
    print("初始状态已保存：", save_path)

    # 二、读取刚保存的初始状态。
    initial_state = torch.load(save_path, weights_only=True)

    # 三、A组加载这份状态。
    model_a = resnet18(num_classes=5)
    model_a.load_state_dict(initial_state)

    # 四、B组也加载同一份状态。
    model_b = resnet18(num_classes=5)
    model_b.load_state_dict(initial_state)

    # 五、检查两组所有参数和buffer是否相同。
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), name

    print("检查通过：两组初始参数和buffer完全相同")

    # 六、先将模型放到MPS，再分别创建优化器。
    model_a = model_a.to(device)
    model_b = model_b.to(device)

    optimizer_a = torch.optim.Adam(
        model_a.parameters(), lr=0.001
    )
    optimizer_b = torch.optim.Adam(
        model_b.parameters(), lr=0.0003
    )

    print("A组学习率：0.001")
    print("B组学习率：0.0003")
        # 两组使用完全相同的数据和预处理。
    # 本次只比较学习率，先不加入随机增强。
    data_root = "/Users/ersan/Downloads/dataset/flower_5/flower_photos"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    dataset = FlowerData(data_root, transform)

    # 只划分一次，两组共用这份划分。
    train_dataset, val_dataset, unused_dataset = random_split(
        dataset,
        [64, 128, len(dataset) - 64 - 128],
        generator=torch.Generator().manual_seed(42)
    )

    epochs = 10
    save_dir = Path(__file__).resolve().parent

    # A组训练，不会改变B组模型的参数。
    run_experiment(
        model_a, optimizer_a,
        train_dataset, val_dataset,
        device, epochs, "A", save_dir
    )

    # B组从之前加载的同一份初始状态开始训练。
    run_experiment(
        model_b, optimizer_b,
        train_dataset, val_dataset,
        device, epochs, "B", save_dir
    )