import torch
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model_resnet18 import resnet18

lr=0.001
batch_size=32

def train_one_epoch(model,loss_fn,optimizer,train_loader,device):
    model.train()
    total_loss=0
    correct=0
    total_num=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        pred=model(images)
        loss=loss_fn(pred,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*len(labels)
        total_num+=len(labels)
        correct+=(pred.argmax(dim=1)==labels).sum().item()
    print(f"训练准确率: {correct/total_num:.2%}, 训练损失: {total_loss/total_num:.4f}")

def val_one_epoch(model,val_loader,device):
    model.eval()
    correct=0
    total_num=0
    with torch.no_grad():
        for images,labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)
            pred=model(images)
            total_num+=len(labels)
            correct+=(pred.argmax(dim=1)==labels).sum().item()
    val_acc=correct/total_num
    print(f"验证准确率: {val_acc:.2%}")
    
    return val_acc


# 数据目录就在当前脚本旁边
data_dir = Path(__file__).resolve().parent / "weather_data"

# ImageFolder 读取图片时会转成 RGB
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data=datasets.ImageFolder(data_dir/"train",transform=transform)
val_data=datasets.ImageFolder(data_dir/"val",transform=transform)

train_loader=DataLoader(train_data,shuffle=True,batch_size=batch_size)
val_loader=DataLoader(val_data,shuffle=False,batch_size=batch_size)
print("类别：",train_data.classes)

device=torch.device("mps")
print("训练设备：",device)
model=resnet18(num_classes=len(train_data.classes)).to(device)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)


best_acc=-1
save_dir = Path(__file__).resolve().parent

for epoch in range(15):
    print(f"第 {epoch+1} 轮")
    train_one_epoch(model,loss_fn,optimizer,train_loader,device)
    val_acc=val_one_epoch(model,val_loader,device)
    checkpoint={
        "epoch":epoch+1,
        "lr":lr,
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "batch_size":batch_size,
        "val_acc":val_acc
    }
    torch.save(checkpoint,save_dir/"last.pth")
    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(checkpoint,save_dir/"best.pth")
