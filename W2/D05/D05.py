import torch
import torchvision
from torch.utils.data import random_split,DataLoader,Dataset
from torchvision import transforms
# from torchvision.models import models
import sys
import matplotlib.pyplot as plt
from pathlib import Path
w2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(w2_path))
from D01.D01 import FashionMNISTDataset
from D02.D02 import SimpleNet
from D03.D03 import train_oneloop,check_accuracy


epochs=15
lr=0.001
batch_size=32
model=SimpleNet()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_fn=torch.nn.CrossEntropyLoss()

if __name__=="__main__":
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

    best_acc = -1
    save_dir = Path(__file__).resolve().parent

    #继续中断前epoch的训练
    start_epoch=0
    last_path=save_dir/"last.pth"
    best_path=save_dir/"best.pth"
    if last_path.exists():
        saved=torch.load(last_path,weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch=saved["epoch"]
        # 已经完成的轮数
        print("已恢复，接着训练第", start_epoch + 1, "轮")
    if best_path.exists():
        best_saved = torch.load(best_path, weights_only=True)
        best_acc = best_saved["val_acc"]

    for epoch in range(start_epoch,epochs):
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
        # 修改：用本轮验证准确率选择best，不使用训练loss或测试集。
        val_acc=correct/total
        acc_log.append(val_acc)
        checkpoint={
            "epoch":epoch+1,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "lr":lr,
            "batch_size":batch_size,
            "val_acc":val_acc
        }

        # 修改：每轮覆盖last.pth，所以它始终是最后一轮。
        torch.save(checkpoint,save_dir/"last.pth")

        # 修改：验证准确率刷新最高值时，才覆盖best.pth。
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(checkpoint,save_dir/"best.pth")
            print("更新best，验证准确率：",best_acc)

    # 修改：torch.load读取文件；load_state_dict将其中的模型参数装入新模型。
    checkpoint_path=save_dir/"best.pth"
    saved_checkpoint=torch.load(checkpoint_path,weights_only=True)
    new_model=SimpleNet()
    new_model.load_state_dict(saved_checkpoint["model"])
    new_model.eval()
    print("已加载best，来自第",saved_checkpoint["epoch"],"轮")
    
