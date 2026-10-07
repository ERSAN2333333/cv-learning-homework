"""垃圾分类 Dataset；保留原有 trashdataset 类名，方便现有代码导入。"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# 【修改】集中定义类别顺序；训练、验证、测试都使用同一套标签编号。
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class trashdataset(Dataset):
    # 【修改】transform 默认可为空，方便单独检查原始图片。
    def __init__(self, root, transform=None):
        self.transform = transform
        self.name_list = CLASS_NAMES.copy()
        self.image_path = []
        self.label = []
        root = Path(root)
        for label, name in enumerate(self.name_list):
            folder = root / name
            # 【修改】路径写错时尽早报错，避免等到训练过程中才发现。
            if not folder.is_dir():
                raise FileNotFoundError(f"缺少类别目录：{folder}")
            # 【修改】只读取图片文件，跳过 .DS_Store、文本文件和子目录。
            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            # 【修改】三个集合中每个类别都应有图片。
            if not files:
                raise ValueError(f"类别目录没有图片：{folder}")
            self.image_path.extend(files)
            self.label.extend([label] * len(files))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # 【修改】用 with 及时关闭原文件；保留 RGB 转换，统一为三通道。
        with Image.open(self.image_path[index]) as original:
            image = original.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label[index]
