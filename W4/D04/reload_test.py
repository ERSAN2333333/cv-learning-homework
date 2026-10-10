"""【新增】新进程加载权重，用已保存的固定输入核验输出。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from W3.D06.model_resnet50 import ResNet50


if __name__ == "__main__":
    save_dir = Path(__file__).resolve().parent
    for name in ("best", "last"):
        # 只读取本练习自己保存的检查点。
        checkpoint = torch.load(
            save_dir / f"{name}.pth", map_location="cpu", weights_only=False
        )
        model = ResNet50(num_classes=len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        with torch.no_grad():
            logits = model(checkpoint["fixed_images"])

        before = checkpoint["reference_logits"]
        max_diff = (logits - before).abs().max().item()
        same_class = torch.equal(logits.argmax(1), before.argmax(1))
        result = f"{name}：重载输出最大差={max_diff:.8g}，预测类别一致={same_class}\n"
        print(result, end="")
        with (save_dir / "train_log.txt").open("a", encoding="utf-8") as file:
            file.write(result)
        assert torch.allclose(logits, before, rtol=1e-5, atol=1e-6), "重载输出不一致"
        assert same_class, "重载后预测类别改变"

    with (save_dir / "train_log.txt").open("a", encoding="utf-8") as file:
        file.write("核验通过：新进程重载best和last后，输出分别与保存前一致。\n")
    print("核验通过，结果已写入train_log.txt。")
