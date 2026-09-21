import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn

def mean_score(item):
    if len(item)==0:
        return ValueError("分数列表不能为空")
    return sum(item)/len(item)

item=[1,2,3,4,5]
item=[]
print("均值:",mean_score(item))

class SimpelData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

class Simpelmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,2)
    def forward(self,x):
        print("输入形状:",x.shape)
        output=self.linear(x)
        print("输出形状",output.shape)
        return output

data=torch.rand(4,3)
label=torch.tensor([0,1,0,1])
traindata=SimpelData(data,label)
trainloder=DataLoader(traindata,shuffle=False,batch_size=2)

model=Simpelmodel()
loss_fn=nn.CrossEntropyLoss()

for x,y in trainloder:
    pred=model(x)
    loss=loss_fn(pred,y)
    print("损失值：",loss)


class Guangpu(Dataset):
    def __init__(self):
        if len(data) != len(label):
            raise ValueError("光谱数量和标签数量不一致")
        self.data=torch.rand(4,8)   ## 4条光谱，每条光谱有8个波段
        self.label=torch.tensor([1,2,3,4])
    def __getitem__(self, index):
        if index>=len(self.data):
            raise IndexError("越界")
        elif index < -len(self):
            raise IndexError("负索引")      #索引-1是最后一个样本，-2是倒数第二个样本
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

dataset=Guangpu()
u,v=dataset[0]
print("第一个样本：", u, v)

# 负索引，返回最后一个样本
u,v = dataset[-1]
print("最后一个样本：", u,v)

# 越界检查
try:
    dataset[4]
except IndexError as error:
    print("捕获越界错误：", error)
