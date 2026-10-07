"""直接运行即可划分数据集，生成 train、val、test 目录。"""
import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# 【新增】固定类别顺序，与 dataset.py 的编号 0～5 一致。
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/Users/ersan/Downloads/dataset/dataset-resized"))
    parser.add_argument("--output", type=Path,
                        default=Path("/Users/ersan/Downloads/dataset/dataset-split"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("原始目录和输出目录不能相同或互相包含")

    if output.exists():
        raise FileExistsError(f"输出目录已存在，请先检查或换一个目录：{output}")

    # 【新增】先检查全部图片；检查成功后才创建输出目录。
    files_by_class = {}
    by_hash = defaultdict(list)
    for name in CLASSES:
        folder = source / name
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少类别目录：{folder}")
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        for path in files:
            with Image.open(path) as image:
                image.verify()
            by_hash[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
        files_by_class[name] = files

    # 【新增】SHA256 相同代表文件内容完全一致。
    # 当前数据有 3 对跨类别重复图片，暂时全部排除这 6 张，保留原文件。
    # 其他重复内容也按相同规则排除，避免进入不同集合造成数据泄漏。
    duplicate_groups = [paths for paths in by_hash.values() if len(paths) > 1]
    excluded = {p for group in duplicate_groups for p in group}
    # 【新增】每类分别按约 70% / 15% / 15% 分层划分。
    # 先排序再固定随机种子；源文件列表不变时可得到相同划分。
    rng = random.Random(args.seed)
    splits = {name: {} for name in ("train", "val", "test")}
    for name in CLASSES:
        files = [p for p in files_by_class[name] if p not in excluded]
        rng.shuffle(files)
        n_train, n_val = int(len(files) * 0.70), int(len(files) * 0.15)
        parts = (files[:n_train], files[n_train:n_train + n_val],
                 files[n_train + n_val:])
        if any(not part for part in parts):
            raise ValueError(f"{name} 的样本不足，无法分成三个非空集合")
        for split, part in zip(splits, parts):
            splits[split][name] = [str(p.relative_to(source)) for p in part]
        print(f"{name}: train={len(parts[0])}, val={len(parts[1])}, test={len(parts[2])}")

    for group in duplicate_groups:
        print("排除重复内容：", [str(p.relative_to(source)) for p in group])
    for split, mapping in splits.items():
        print(f"{split} 合计：{sum(len(files) for files in mapping.values())}")
    manifest = {
        "source": str(source), "seed": args.seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "classes": CLASSES, "splits": splits,
        "excluded_duplicate_groups": [
            [str(p.relative_to(source)) for p in group] for group in duplicate_groups
        ],
    }
    # 【修改】直接运行就复制图片并生成划分结果，无需额外传入执行参数。
    output.mkdir(parents=True, exist_ok=False)
    for split, mapping in splits.items():
        for name, files in mapping.items():
            (output / split / name).mkdir(parents=True)
            for relative in files:
                shutil.copy2(source / relative, output / split / relative)
    # 【新增】保存实际文件清单，便于检查集合交集、类别顺序及复现实验。
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"划分完成：{output}")


if __name__ == "__main__":
    main()
