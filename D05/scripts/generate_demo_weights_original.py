"""生成两个可追溯的教学权重；只用于Git/实验记录练习。"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_data():
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(96, 3, generator=generator)
    # 固定规则产生标签，使两次实验使用完全相同的数据。
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2] > 0).long()
    return x[:64], y[:64], x[64:], y[64:]


def make_model():
    return nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def train(run_id, lr):
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "run_id": run_id,
        "purpose": "教学用合成二分类；只改变学习率",
        "model": "Linear(3,8)-ReLU-Linear(8,2)",
        "lr": lr,
        "batch_size": 16,
        "epochs": 8,
        "seed": 42,
        "data_seed": 2026,
        "selection": "验证集准确率最高；并列取最早轮",
        "code_file": "generate_demo_weights.py",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.manual_seed(config["seed"])
    train_x, train_y, val_x, val_y = make_data()
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    best_epoch = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(
            len(train_x), generator=torch.Generator().manual_seed(1000 + epoch)
        )
        total_loss = 0.0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start : start + config["batch_size"]]
            scores = model(train_x[indices])
            loss = loss_fn(scores, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)

        model.eval()
        with torch.no_grad():
            val_scores = model(val_x)
            val_accuracy = float((val_scores.argmax(1) == val_y).float().mean())

        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_x),
            "val_accuracy": val_accuracy,
        }
        history.append(record)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pth")

    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    weight_hash = sha256(run_dir / "best.pth")
    record = (
        f"# {run_id}\n\n"
        f"- 目的：合成数据下比较学习率；不能作为真实分类结果。\n"
        f"- 学习率：{lr}\n"
        f"- 代码：generate_demo_weights.py\n"
        f"- 配置：runs/{run_id}/config.json\n"
        f"- 权重：runs/{run_id}/best.pth\n"
        f"- 最佳轮数：{best_epoch}\n"
        f"- 最佳验证准确率：{best_accuracy:.6f}\n"
        f"- 权重SHA256：{weight_hash}\n"
        f"- 不确定项：仅一次种子、合成小数据，不能判断真实任务优劣。\n"
    )
    (run_dir / "实验记录.md").write_text(record, encoding="utf-8")
    return run_dir, weight_hash


if __name__ == "__main__":
    runs_root = ROOT / "runs"
    if runs_root.exists():
        raise FileExistsError("runs目录已存在；为防止覆盖，请先检查已有结果")

    results = [
        train("exp001_lr001", 0.001),
        train("exp002_lr0003", 0.0003),
    ]
    for run_dir, weight_hash in results:
        print(run_dir)
        print("SHA256:", weight_hash)

