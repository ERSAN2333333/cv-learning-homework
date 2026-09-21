import torch
import sys
from pathlib import Path
import json
import torchvision

print("当前工作目录:",Path.cwd())
print("脚本位置；",Path(__file__).resolve())
print("脚本所在目录：", Path(__file__).resolve().parent)
print("python解释器路径：",sys.executable)


config={
    "lr":0.01,
    "batch_size":4
}

#写入JSON
with open("day04.json","w",encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# 读取JSON
with open("day04.json", "r", encoding="utf-8") as file:
    loaded_config = json.load(file)

print("原始配置：", config)
print("读取配置：", loaded_config)

print("lr是否一致：", config["lr"] == loaded_config["lr"])
print(
    "batch_size是否一致：",
    config["batch_size"] == loaded_config["batch_size"]
)

assert config == loaded_config
print("配置读写一致")

print("----------------------------------")
print("Pytorch版本：",torch.__version__)
print("Pytorchvision版本:",torchvision.__version__)
print("PyTorch对应CUDA版本：", torch.version.cuda)