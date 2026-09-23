import struct

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# 一、读取Fashion-MNIST原始文件

data_root = "/Users/ersan/Downloads/Fashion-MNIST/raw"


def read_images(file_path):
    with open(file_path, "rb") as file:
        _, image_count, height, width = struct.unpack(">IIII", file.read(16))
        images = torch.frombuffer(bytearray(file.read()), dtype=torch.uint8)
        images = images.reshape(image_count, height, width)
    return images


def read_labels(file_path):
    with open(file_path, "rb") as file:
        _, label_count = struct.unpack(">II", file.read(8))
        labels = torch.frombuffer(bytearray(file.read()), dtype=torch.uint8)
        labels = labels.long()

    assert len(labels) == label_count
    return labels


# 二、定义自己的Dataset

class FashionMNISTDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.images = read_images(image_path)
        self.labels = read_labels(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index].numpy())
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # 三、创建数据集并划分

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    all_train_dataset = FashionMNISTDataset(
        data_root + "/train-images-idx3-ubyte",
        data_root + "/train-labels-idx1-ubyte",
        transform
    )

    unused_count = len(all_train_dataset) - 2000 - 500

    train_dataset, val_dataset, unused_dataset = random_split(
        all_train_dataset,
        [2000, 500, unused_count],
        generator=torch.Generator().manual_seed(42)
    )

    torch.save(
        {
            "train": train_dataset.indices,
            "val": val_dataset.indices
        },
        "/Users/ersan/rs_cvpy/cv_learning/W2/D01/split_indices.pt"
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    # 四、检查数据

    images, labels = next(iter(train_loader))

    print("批次形状：", images.shape)
    print("输入最小值：", images.min().item())
    print("输入最大值：", images.max().item())
    print("标签类型：", labels.dtype)

    assert images.min() >= -1
    assert images.max() <= 1
    assert labels.dtype == torch.int64

    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    same_indices = train_indices.intersection(val_indices)

    print("重复索引数量：", len(same_indices))
    assert len(same_indices) == 0

    for batch_images, batch_labels in train_loader:
        train_last_batch = len(batch_labels)

    for batch_images, batch_labels in val_loader:
        val_last_batch = len(batch_labels)

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))
    print("训练集最后一批：", train_last_batch)
    print("验证集最后一批：", val_last_batch)


    # 五、显示16张图片并核对类别名

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    figure, axes = plt.subplots(4, 4, figsize=(8, 8))

    for image, label, axis in zip(images, labels, axes.ravel()):
        image = image * 0.5 + 0.5
        label_number = label.item()

        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(class_names[label_number])
        axis.axis("off")

    plt.tight_layout()
    plt.show()


    test_train_indices=train_indices.copy()
    test_val_indices=val_indices.copy()

    repear_index=train_dataset.indices[0]
    test_val_indices.add(repear_index)
    # 再次检查交集
    test_same_indices = test_train_indices.intersection(
        test_val_indices
    )

    print("故意加入的重复索引：", repear_index)
    print("检测到的重复索引：", test_same_indices)

    assert len(test_same_indices) == 0, "检测到训练集和验证集存在重复样本"
