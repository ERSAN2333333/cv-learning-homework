import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import struct
import sys
from pathlib import Path
# 把W2文件夹加入搜索路径，Python就能找到旁边的D01文件夹。
w2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(w2_path))
from D01.D01 import FashionMNISTDataset
from D02.D02 import SimpleNet


model=SimpleNet()

batch_size=32
epochs=10
learning_rate=0.001
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# 示例：只取一个批次，观察一次step前后参数（下面默认不调用）
def compare_one_batch_step(trainloader,model,loss_fn,optimizer):
    model.train()
    data,label=next(iter(trainloader))  # 只取一个真实批次
    old_weight=model.block[0].weight.detach().clone()

    optimizer.zero_grad()
    pred=model(data)
    # print(model.grad)
    # pred = pred.detach(). #使用detach截断梯度传播，反向传播backward会报错
    loss=loss_fn(pred,label)
    print("pred需要梯度吗：", pred.requires_grad)
    print("loss需要梯度吗：", loss.requires_grad)
    loss.backward()
    print("第一层有梯度吗：", model.block[0].weight.grad is not None)
    optimizer.step()

    new_weight=model.block[0].weight.detach()
    print("更新前的一个参数：",old_weight[0,0].item())
    print("更新后的同一个参数：",new_weight[0,0].item())
    print("第一层有参数变化：",not torch.equal(old_weight,new_weight))
    print("loss：",loss.item())

# 完整训练一轮：依次训练trainloader里的每个批次
def train_oneloop(trainloader,model,loss_fn,optimizer):
    model.train()
    total_loss=0
    total_samples=0
    correct=0
    for data,label in trainloader:
        optimizer.zero_grad()
        pred=model(data)
        predicted_labels=pred.argmax(dim=1)

        loss=loss_fn(pred,label)
        loss.backward()
        optimizer.step()
        # loss.item()是本批平均值，乘本批人数后才能汇总整轮。
        total_loss+=loss.item()*len(label)
        total_samples+=len(label)
        correct+=(predicted_labels==label).sum().item()
    print("本轮平均loss：",total_loss/total_samples)
    print("本轮平均acc：",correct/total_samples)

# 验证准确率：预测类别等于真实标签的比例
def check_accuracy(valloader,model):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data,label in valloader:
            pred=model(data)
            predicted_labels=pred.argmax(dim=1)
            correct+=(predicted_labels==label).sum().item()
            total+=len(label)
    print(f"验证准确率：{correct/total*100:.1f}%")

# 训练一个批次会更新权重；验证一个批次只做前向计算。
def compare_train_and_val(trainloader,valloader,model,loss_fn,optimizer):
    model.train()
    train_data,train_label=next(iter(trainloader))
    before_train=model.block[0].weight.detach().clone()

    optimizer.zero_grad()
    train_pred=model(train_data)
    train_loss=loss_fn(train_pred,train_label)
    train_loss.backward()
    optimizer.step()

    after_train=model.block[0].weight.detach().clone()
    print("训练后参数变化：",not torch.equal(before_train,after_train))

    model.eval()
    val_data,val_label=next(iter(valloader))
    before_val=model.block[0].weight.detach().clone()
    with torch.no_grad():
        val_pred=model(val_data)
        val_loss=loss_fn(val_pred,val_label)
    after_val=model.block[0].weight.detach().clone()

    print("验证后参数变化：",not torch.equal(before_val,after_val))
    print("验证批次loss：",val_loss.item())

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

    

    #----------------2.连续新建图反传，观察梯度累加--------------------
    w = torch.tensor(2.0, requires_grad=True)
    # 第一次计算：建立第一张图
    loss1=(3*w-1)**2
    loss1.backward()
    print("第一次反传后：", w.grad.item())  # 30

    # 重新计算：建立第二张图
    loss2 = (3 * w - 1) ** 2
    loss2.backward()
    print("第二次反传后：", w.grad.item())  # 60
    #两次都使用同一个w，而w.grad没有清零，因此第二次看到的是30 + 30 = 60。

    #------------3.只取一批，比较一次step前后参数。需要时取消下一行注释。------------
    # compare_one_batch_step(train_loader, model, loss_fn, optimizer)

    # 完整训练：需要时再取消下面几行注释。
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_oneloop(train_loader, model, loss_fn, optimizer)
        check_accuracy(val_loader, model)
    # compare_one_batch_step(train_loader,model,loss_fn,optimizer)
    # compare_train_and_val(train_loader,val_loader,model,loss_fn,optimizer)
