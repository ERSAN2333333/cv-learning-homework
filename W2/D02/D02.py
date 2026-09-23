import torch
import torchvision
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
import sys
from pathlib import Path

# 把W2文件夹加入搜索路径，Python就能找到旁边的D01文件夹。
w2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(w2_path))
from D01.D01 import FashionMNISTDataset
import torch.nn as nn
import struct



class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.block=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        
        x=self.flatten(x)

        x=self.block(x)
        return x


if __name__ == "__main__":
    # 一、读取Fashion-MNIST原始文件

    data_root = "/Users/ersan/Downloads/Fashion-MNIST/raw"


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

    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)

    model=SimpleNet()
    Loss=torch.nn.CrossEntropyLoss()

    for image,label in train_loader:
        pred=model(image)
        # print(pred)
        run_loss=Loss(pred,label)

    print("损失:",run_loss.item())


    #————————————————2.手算小矩阵乘法，检查线性层权重轴。————————————————————————
    x=torch.tensor([[1.0,2.0]])
    print(x.shape)    #(1,2)
    weight=torch.tensor([
        [1.0,0.0],
        [0.0,1.0],
        [1.0,1.0]
    ])
    bias=torch.tensor([1.0,1.0,1.0])
    print(weight.shape)
    print(bias.shape)
    mau_result=x@weight.T+bias
    print("矩阵计算结果：",mau_result)
    # 手算：
    # 第一个输出：1×1 + 2×0 + 1 = 2
    # 第二个输出：1×0 + 2×1 + 1 = 3
    # 第三个输出：1×1 + 2×1 + 1 = 4

    linear=nn.Linear(2,3)

    # 把指定的权重和偏置放入线性层。
    # 这里只设置数值，不需要记录梯度。
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    linear_result = linear(x)
    print("线性层计算结果：", linear_result)

    assert torch.allclose(mau_result, linear_result)


    #————————————————3.比较正确类logit提高前后的交叉熵————————————————————
    correct_label=torch.tensor([2])           #正确标签类别是2
    # correct_label=torch.tensor([1.5])      #错误类别是1.5

    #三个类别原本的分数
    logits_before=torch.tensor([[0.0,0.0,1.0]])
    # 只提高正确类别（编号2）的分数，其他两个不变
    logits_after = torch.tensor([[0.0, 0.0, 3.0]])

    loss_fn=torch.nn.CrossEntropyLoss()
    loss_before=loss_fn(logits_before,correct_label)
    loss_after=loss_fn(logits_after,correct_label)
    print("提高前的交叉熵：", loss_before.item())
    print("提高后的交叉熵：", loss_after.item())

    assert loss_after < loss_before



    #————————————————————-补练-----------------------
    # 原始标签是1～9
    original_labels = torch.tensor([1, 5, 9])

    # 9类模型输出：每个样本有9个类别分数
    logits = torch.randn(3, 9)
    loss_fn = nn.CrossEntropyLoss()

    # 正向映射：原标签1～9 → 模型标签0～8
    model_labels = original_labels - 1
    print("模型标签：", model_labels)  # tensor([0, 4, 8])

    loss = loss_fn(logits, model_labels)
    print("损失：", loss.item())

    # 反向映射：模型标签0～8 → 原标签1～9
    pred_model_labels = logits.argmax(dim=1)
    pred_original_labels = pred_model_labels + 1

    print("模型预测编号：", pred_model_labels)
    print("对应的原始标签：", pred_original_labels)
