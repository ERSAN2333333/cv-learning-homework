import torch
import sys
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
w2_path=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(w2_path))
from W3.D06.dataset import trashdataset
from W3.D06.model_resnet50 import ResNet50
from torchvision.models import ResNet50_Weights,resnet50
from matplotlib import pyplot

torch.manual_seed(42)

# 【新增】使用Mac的MPS设备。
if not torch.backends.mps.is_available():
    raise RuntimeError("当前Python环境无法使用MPS")
device = torch.device("mps")


data_transforms={"train":transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
            "val":transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        }

train_path="/Users/ersan/Downloads/dataset/dataset-split/train"
val_path="/Users/ersan/Downloads/dataset/dataset-split/val"

train_dataset=trashdataset(train_path,transform=data_transforms["train"])
val_dataset=trashdataset(val_path,transform=data_transforms["val"])

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)

num_classes=len(train_dataset.name_list)
weights=ResNet50_Weights.IMAGENET1K_V2

model=resnet50(weights=weights)
fc_num=model.fc.in_features
model.fc=torch.nn.Linear(fc_num,num_classes)
model.requires_grad_(False)
model.fc.requires_grad_(True)
model = model.to(device)  # 【新增】创建优化器前，将整个模型移到MPS。



epochs=2
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.fc.parameters(),lr=1e-3)


#最后一轮预测错误的图片
false_dir = Path(__file__).resolve().parent / "false_pred"
false_dir.mkdir(exist_ok=True)
false_count = 0
# 用于还原归一化后的图片颜色
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# 【修改】分别记录每轮训练、验证的平均loss和准确率。
train_loss_list=[]
train_acc_list=[]
loss_list=[]
epoch_list=[]
acc_list=[]
for epoch in range(epochs):
    model.eval()
    model.fc.train()
    total_loss=0
    total_num=0
    train_correct=0  # 【新增】训练集预测正确的数量
    for image,label in train_loader:
        # 【新增】模型、图片和标签使用同一个设备。
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        logits=model(image)
        loss=loss_fn(logits,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*len(label)
        total_num+=len(label)
        train_correct+=(logits.argmax(dim=1)==label).sum().item()
    train_loss_list.append(total_loss/total_num)
    train_acc_list.append(train_correct/total_num*100)
    print(f"第{epoch + 1}轮，训练loss：{total_loss / total_num:.4f}")

    model.eval()
    total_loss=0
    total_num=0
    correct=0
    # 【新增】每轮重新统计，循环结束后保留最后一轮验证的混淆矩阵。
    confusion=torch.zeros(num_classes,num_classes,dtype=torch.long)
    with torch.no_grad():
        for image,label in val_loader:
            image, label = image.to(device), label.to(device)  # 【新增】验证也使用MPS
            logits=model(image)
            loss=loss_fn(logits,label)
            total_loss+=loss.item()*len(label)
            total_num+=len(label)
            logits_label=logits.argmax(dim=1)
            correct+=(logits_label==label).sum().item()
            # 【新增】行是真实类别，列是预测类别。
            # 【修改】转回CPU统计，混淆矩阵继续留在CPU供绘图使用。
            for i, (true_label, pred_label) in enumerate(zip(label.cpu().tolist(), logits_label.cpu().tolist())):
                confusion[true_label,pred_label]+=1

                # 只保存最后一轮预测错误的图片
                if epoch == epochs - 1 and true_label != pred_label:
                    false_count += 1

                    # 转回CPU，反归一化，再转换为绘图需要的HWC排列
                    picture = image[i].cpu() * std + mean
                    picture = picture.clamp(0, 1).permute(1, 2, 0).numpy()

                    fig_error, ax_error = pyplot.subplots(figsize=(4, 4))
                    ax_error.imshow(picture)
                    ax_error.set_title(
                        f"True: {val_dataset.name_list[true_label]}\n"
                        f"Pred: {val_dataset.name_list[pred_label]}"
                    )
                    ax_error.axis("off")
                    fig_error.tight_layout()
                    # 【修改】明确PNG格式，保存带有真实/预测标签的整张图。
                    output_path = false_dir / f"false_{false_count:03d}.png"
                    fig_error.savefig(output_path, format="png", bbox_inches="tight")
                    pyplot.close(fig_error)
    print(f"当前第{epoch+1}轮,acc:{correct/total_num},loss:{total_loss/total_num}")
    # 【修改】保存整轮平均loss，用1开始的轮数作横轴。
    loss_list.append(total_loss/total_num)
    epoch_list.append(epoch+1)
    acc_list.append(correct/total_num*100)


# 【补回】训练结束后显示实际保存数量和完整路径。
print(f"最后一轮共保存 {false_count} 张错分图片，目录：{false_dir}", flush=True)

# 【新增】训练和验证曲线；只显示，不额外保存文件。
fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epoch_list, train_loss_list, "o-", label="Train")
axes[0].plot(epoch_list, loss_list, "o-", label="Validation")
axes[0].set_ylabel("Loss")
axes[0].set_title("Average loss")
axes[1].plot(epoch_list, train_acc_list, "o-", label="Train")
axes[1].plot(epoch_list, acc_list, "o-", label="Validation")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_xticks(epoch_list)
    ax.legend()
fig.tight_layout()

# 【新增】最后一轮验证集的混淆矩阵，数字表示样本数量。
fig, ax = pyplot.subplots(figsize=(7, 6))
heatmap = ax.imshow(confusion.numpy(), cmap="Blues")
fig.colorbar(heatmap, ax=ax)
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(train_dataset.name_list, rotation=45, ha="right")
ax.set_yticklabels(train_dataset.name_list)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Validation confusion matrix (last epoch)")
for row in range(num_classes):
    for col in range(num_classes):
        count = confusion[row, col].item()
        color = "white" if count > confusion.max().item()/2 else "black"
        ax.text(col, row, str(count), ha="center", va="center", color=color)
fig.tight_layout()
pyplot.show()


