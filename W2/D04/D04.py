import torch
import torchvision
import struct
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import sys
from pathlib import Path
w2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(w2_path))
from D01.D01 import FashionMNISTDataset
from D02.D02 import SimpleNet
from D03.D03 import train_oneloop,check_accuracy
import matplotlib.pyplot as plt

model=SimpleNet()
epochs=3
lr=0.001
batch_size=32
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

if __name__=="__main__":
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

    loss_log=[]
    epoch_log=[]
    acc_log=[]
    for epoch in range(epochs):
        print(f"第{epoch+1}轮")
        # train_oneloop(train_loader,model,loss_fn,optimizer)
        # check_accuracy(val_loader,model)
        
        total_loss=0
        total_sample=0
        model.train()
        for data,label in train_loader:
            optimizer.zero_grad()
            pred=model(data)
            loss=loss_fn(pred,label)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(label)
            total_sample+=len(label)
        print("loss:",total_loss/total_sample)
        loss_log.append(total_loss / total_sample)
        epoch_log.append(epoch+1)

        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for data,label in val_loader:
                pred=model(data)
                predict_label=pred.argmax(dim=1)
                correct+=(predict_label==label).sum().item()
                total+=len(data)
            print("acc:",correct/total)
        acc_log.append(correct/total)

    fig,axes=plt.subplots(1,2,figsize=(8,5))
    axes[0].plot(epoch_log,loss_log,color="b",label="Training loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('loss-epoch')
    axes[1].plot(epoch_log,acc_log,color="b",label="valing acc")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('acc')
    axes[1].set_title('acc-epoch')
    plt.tight_layout()
    output_dir=Path(__file__).parent
    fig.savefig(output_dir/"training_curves.png")

    with open(output_dir/"three_epoch_log.txt","w",encoding="utf-8") as file:
        for epoch_number,train_loss,val_acc in zip(epoch_log,loss_log,acc_log):
            file.write(f"第{epoch_number}轮：训练平均loss={train_loss:.4f}，验证准确率={val_acc*100:.1f}%\n")

    # 故意让训练、验证索引重复一次，记录断言发现错误的结果。
    test_train=set(train_dataset.indices)
    test_val=set(val_dataset.indices)
    test_val.add(train_dataset.indices[0])
    try:
        assert len(test_train.intersection(test_val))==0, "检测到训练和验证样本重叠"
    except AssertionError as error:
        with open(output_dir/"failed_check.txt","w",encoding="utf-8") as file:
            file.write(str(error)+"\n")

    print("日志、曲线和失败检查记录已保存到：",output_dir)
    plt.show()
