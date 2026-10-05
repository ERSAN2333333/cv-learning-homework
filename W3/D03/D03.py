import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset, random_split, Subset
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import os


class FlowerData(Dataset):
    def __init__(self,root,transform=None):
        self.transform=transform
        self.class_name=[
            "daisy",
            "dandelion",
            "roses",
            "sunflowers",
            "tulips"
        ]
        self.image_paths=[]
        self.labels=[]
        
        # 依次进入5个类别文件夹
        for label in range(len(self.class_name)):
            class_name=self.class_name[label]
            folder_path=Path(root)/class_name

            # sorted让文件读取顺序固定
            for file_name in sorted(os.listdir(folder_path)):
                image_path=os.path.join(folder_path,file_name)
                self.image_paths.append(image_path)
                self.labels.append(label)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image_path=self.image_paths[index]
        image=Image.open(image_path).convert('RGB')
        label=self.labels[index]
        if self.transform is not None:
            image=self.transform(image)
        return image,label


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def _make_layer(self, out_channels, block_count, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [
            BasicBlock(self.in_channels, out_channels, stride, downsample)
        ]
        self.in_channels = out_channels

        for _ in range(1, block_count):
            blocks.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes):
    return ResNet18(num_classes)

if __name__=="__main__":
    data_root="/Users/ersan/Downloads/dataset/flower_5/flower_photos"
    # 修改：使用Mac的MPS设备训练。
    device = torch.device("mps")
    print("训练设备：", device)

    # 修改：先用一种增强做对照。无增强实验改为0.0，增强实验用0.5。
    # 两次均重新运行脚本，保持划分、初始化、学习率、batch和轮数相同。
    flip_probability = 0.5
    data_transform={
        "train":transforms.Compose([
            # 修改：与验证集使用相同的基础处理，只额外增加随机水平翻转。
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=flip_probability),
            transforms.ToTensor(),
            # 修改：Normalize需要mean和std；这里将[0,1]映射到[-1,1]。
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        "val":transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    }

    # 修改：传入具体的Compose，而不是transforms模块。
    dataset=FlowerData(data_root,data_transform["train"])
    # 只划分一次：64张训练、128张验证，其余暂时不用。
    # 两个集合没有重复的图片索引。
    train_dataset, val_dataset, unused_dataset = random_split(
        dataset,
        [64, 128, len(dataset) - 64 - 128],
        generator=torch.Generator().manual_seed(42)
    )

    # 修改：random_split得到的子集共享同一个dataset。
    # 另外创建验证来源，并沿用刚才的验证索引，才能单独使用验证预处理。
    # FlowerData按固定类别和排序后的文件读取，所以两个来源的索引一一对应。
    val_source = FlowerData(data_root, data_transform["val"])
    val_dataset = Subset(val_source, val_dataset.indices)

    # 第四点【只阅读，不执行】：过拟合时，后续可以单独扩大训练集。
    # 本次64张对照实验先不改。下面从未使用的图片中补192张，变成256张训练。
    # 保留原来的128张验证图片不动，避免换了验证集导致结果不好比较。
    # expanded_indices = train_dataset.indices + unused_dataset.indices[:192]
    # train_dataset = Subset(dataset, expanded_indices)
    # 注意：扩大数据后，相同轮数会有更多更新步数，需要在实验记录中说明。

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        # 修改：打乱顺序使用独立的随机种子，不受随机增强消耗随机数影响。
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # 修改：每次从同一初始化开始，方便比较有无增强。
    torch.manual_seed(42)
    # 修改：先把模型移到MPS，再创建优化器。
    model=resnet18(num_classes=5).to(device)
    epochs=10
    lr=0.001
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=torch.nn.CrossEntropyLoss()

    # 第四点【只阅读，不执行】：每次只选一个动作，重新运行实验作比较。
    # 学习不足：先确认标签、预处理、梯度和参数更新正常。
    # 若训练loss仍在下降，才考虑延长训练；过拟合时不要优先增加轮数。
    # epochs = 20
    # 若loss震荡明显，可以单独尝试更小的学习率，不是所有问题都需要降低lr。
    # lr = 0.0003
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 过拟合：也可以单独尝试权重衰减，限制权重过大；不保证一定改善。
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    # 水平翻转对照已有flip_probability：分别用0.0和0.5，其余设置不变。

    # 第四点【只阅读，不执行】：早停准备，需配合循环末尾的早停示例。
    # best_val_loss = float("inf")
    # patience = 3  # 连续3轮验证loss没有刷新最好结果，就停止。
    # bad_epochs = 0

    # 分别记录每轮的训练、验证结果
    train_loss_log = []
    val_loss_log = []
    train_acc_log = []
    val_acc_log = []
    for epoch in range(epochs):
        model.train()
        total_loss=0
        total_num=0
        correct=0
        for image,label in train_loader:
            # 修改：训练图片和标签都放到MPS，与模型在同一个设备。
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()   
            pred=model(image)
            loss=loss_fn(pred,label)
            # 第四点【只阅读，不执行】：更新前复制权重，用于排查学习不足。
            # old_weight = model.fc.weight.detach().clone()
            loss.backward()
            # 第四点【只阅读，不执行】：反传后看第一层有没有梯度。
            # print("第一层有梯度：", model.stem[0].weight.grad is not None)
            optimizer.step()
            # print("分类头权重有变化：", not torch.equal(old_weight, model.fc.weight))
            # 有梯度不代表参数一定更新；这里只抽查两处，不代表所有层都正常。
            total_num+=len(label)
            total_loss+=loss.item()*len(label)
            # 每张图片分别选出分数最高的类别
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == label).sum().item()
        train_loss = total_loss / total_num
        train_acc = correct / total_num
        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)

        model.eval()
        total_loss = 0
        total_num = 0
        correct = 0
        with torch.no_grad():
            for image,label in val_loader:
                # 修改：验证图片和标签也需要放到MPS。
                image = image.to(device)
                label = label.to(device)
                pred=model(image)
                loss=loss_fn(pred,label)
                total_num+=len(label)
                # 修改：累加每个批次的loss总和，不能每批覆盖前面的结果。
                total_loss+=loss.item()*len(label)
                correct+=(pred.argmax(dim=1)==label).sum().item()
        val_loss = total_loss / total_num
        val_acc = correct / total_num

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"第{epoch + 1}轮")
        print(f"训练loss：{train_loss:.4f}，准确率：{train_acc:.2%}")
        print(f"验证loss：{val_loss:.4f}，准确率：{val_acc:.2%}")

        # 第四点【只阅读，不执行】：过拟合时，用验证loss保存最佳状态并早停。
        # 只看验证集选择模型，不用测试集；需同时启用前面的早停准备代码。
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     bad_epochs = 0
        #     torch.save(model.state_dict(), Path(__file__).parent / "best_early_stop.pth")
        # else:
        #     bad_epochs += 1
        # if bad_epochs >= patience:
        #     print("验证loss连续3轮没有改善，停止训练")
        #     break
        # 停止时内存中是最后一轮模型；推理前还要加载保存的最佳权重并调用eval()。

    # 新增：展示同一张训练图片翻转前后，人工确认花朵类别仍然正确。
    # 为看清效果，这里固定展示一次翻转；训练仍按flip_probability随机翻转。
    sample_index = train_dataset.indices[0]
    sample_path = dataset.image_paths[sample_index]
    sample_label = dataset.labels[sample_index]
    sample_image = Image.open(sample_path).convert("RGB")
    original_image = data_transform["val"](sample_image)
    # 图片张量为[C,H,W]，沿宽度轴翻转就是左右翻转。
    flipped_image = original_image.flip(dims=[2])
    class_name = dataset.class_name[sample_label]

    preview_fig, preview_axes = plt.subplots(1, 2, figsize=(7, 3))
    # 显示前把[-1,1]恢复到[0,1]，并把CHW改为HWC。
    preview_axes[0].imshow((original_image * 0.5 + 0.5).permute(1, 2, 0))
    preview_axes[0].set_title(f"Before flip: {class_name} ({sample_label})")
    preview_axes[0].axis("off")
    preview_axes[1].imshow((flipped_image * 0.5 + 0.5).permute(1, 2, 0))
    preview_axes[1].set_title(f"After flip: {class_name} ({sample_label})")
    preview_axes[1].axis("off")
    preview_fig.tight_layout()

    # ---------- 画曲线 ----------
    epoch_log = list(range(1, epochs + 1))
    # 第四点【只阅读，不执行】：如果启用早停，用实际完成的轮数代替上一行。
    # epoch_log = list(range(1, len(train_loss_log) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epoch_log, train_loss_log, label="Train loss")
    axes[0].plot(epoch_log, val_loss_log, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epoch_log, train_acc_log, label="Train accuracy")
    axes[1].plot(epoch_log, val_acc_log, label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
