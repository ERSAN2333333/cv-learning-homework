"""标准非分组 ResNet50，可加载 torchvision 官方 ImageNet 权重。"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights


class Bottleneck(nn.Module):
    # 每个残差块：1×1降维 → 3×3提取特征 → 1×1升维。
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        # 不传groups参数，所有卷积使用默认groups=1（不分组）。
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # 与torchvision一致：下采样步长放在3×3卷积，即ResNet V1.5。
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.in_channels = 64
        # 命名与官方一致，方便严格核对并加载预训练权重。
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # 四个阶段包含3、4、6、3个Bottleneck块。
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels, block_count, stride=1):
        out_channels = channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        blocks = [Bottleneck(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, block_count):
            blocks.append(Bottleneck(self.in_channels, channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        print("输入：", x.shape)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        print("maxpool：", x.shape)
        x = self.layer1(x)
        print("layer1：", x.shape)
        x = self.layer2(x)
        print("layer2：", x.shape)
        x = self.layer3(x)
        print("layer3：", x.shape)
        x = self.layer4(x)
        print("layer4：", x.shape)
        x = self.avgpool(x)
        print("avgpool：", x.shape)
        x = torch.flatten(x, 1)
        print("展平：", x.shape)
        # 输出logits，配合CrossEntropyLoss使用，不要提前softmax。
        x = self.fc(x)
        print("分类输出：", x.shape)
        return x

def resnet50(num_classes=6, weights=None):
    """weights=None随机初始化；传入ResNet50_Weights.IMAGENET1K_V2加载预训练。

    首次加载会联网下载约98 MB，并缓存到torch.hub.get_dir()/checkpoints。
    官方权重对应1000类；用于6类时只加载骨干，分类头重新初始化。
    """
    weights = ResNet50_Weights.verify(weights)
    model = ResNet50(num_classes=num_classes)
    if weights is not None:
        state_dict = dict(weights.get_state_dict(progress=True, check_hash=True))
        if num_classes == 1000:
            model.load_state_dict(state_dict, strict=True)
        else:
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            result = model.load_state_dict(state_dict, strict=False)
            if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
                raise RuntimeError(f"预训练权重与模型结构不匹配：{result}")
    return model


# 训练文件中的使用示例（这里只注释，不会下载或启动训练）：
# weights = ResNet50_Weights.IMAGENET1K_V2
# model = resnet50(num_classes=6, weights=weights)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = model.to(device)
# val_transform = weights.transforms()  # 官方配套验证预处理
# 训练增强可以另外定义，但应使用RGB三通道，并使用同样的归一化参数：
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]。
# 你的Dataset读取图片时应保证image.convert("RGB")。



