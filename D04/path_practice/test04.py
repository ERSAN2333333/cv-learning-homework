from pathlib import Path

# print("当前工作目录：",Path.cwd())
# print("当前脚本文件位置；",Path(__file__).resolve())
# print("data路径：",Path("data").resolve())

# data_file = Path("data") / "test04.txt"

# print("寻找文件：", data_file.resolve())
# print("文件存在：", data_file.exists())

# import argparse



# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--data-root",
#     required=True,
#     help="数据根目录"
# )

# args = parser.parse_args()

# # 把传入的路径转换成绝对路径
# data_root = Path(args.data_root).expanduser().resolve()

# # 拼接具体文件路径
# data_file = data_root / "test04.txt"

# print("当前工作目录：", Path.cwd())
# print("脚本位置：", Path(__file__).resolve())
# print("数据根目录：", data_root)
# print("数据文件路径：", data_file)
# print("数据文件是否存在：", data_file.exists())

# if not data_file.exists():
#     raise FileNotFoundError(f"找不到数据文件：{data_file}")

# print("读取内容：", data_file.read_text(encoding="utf-8"))

import argparse
import json
import sys

parser=argparse.ArgumentParser()

config=(
    {
        "lr":0.01,
        "batch_size":8,
        "data-root":"data"
    }
)

# 配置文件始终放在当前脚本旁边
script_dir = Path(__file__).resolve().parent.parent
config_path = script_dir / "homework5.json"
print(config_path)

with open(config_path,"w",encoding="utf-8") as f:
    json.dump(config,f)

with open(config_path,"r",encoding="utf-8") as f:
    json_config=json.load(f)

print(json_config)

# 相对数据目录以配置文件所在目录为基准
data_root = (
    config_path.parent / json_config["data-root"]
).resolve()

data_file = data_root / "homework5.txt"

print("配置文件：", config_path)
print("数据根目录：", data_root)
print("实际查找文件：", data_file)

if not data_file.is_file():
    raise FileNotFoundError(
        f"找不到数据文件：{data_file}；"
        f"请检查配置中的data-root字段"
    )

print("文件路径存在")
print(data_file.read_text(encoding="utf-8"))