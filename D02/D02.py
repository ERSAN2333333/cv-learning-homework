import torch

x=torch.arange(120).reshape(2,3,4,5)
x_b=x[0,:]
x_c=x[:,1,:,:]
print(x_b.shape)
print(x_c.shape)

print(x[0,1,2,3])
x_change=x.permute(0,2,3,1)
print(x_change[0,2,3,1])

x1=torch.rand(3,4)
x2=torch.rand(3,4)

x_s=torch.stack((x1,x2),dim=0)
x_c=torch.cat((x1,x2),dim=0)
x_s1=torch.stack((x1,x2),dim=1)
x_c1=torch.cat((x1,x2),dim=1)
print(x_s.shape)
print(x_c.shape)
print(x_s1.shape)
print(x_c1.shape)

bias=torch.tensor([10.,20.,30.]).reshape(1,3,1,1)  #对通道C操作
z=x+bias
print("x形状：", x.shape)
print("bias形状：", bias.shape)
print("z形状：", z.shape)
print(z[0, 0,0,0] - x[0, 0,0,0])  # 应全部为10
print(z[0, 1] - x[0, 1])  # 应全部为20
print(z[0, 2] - x[0, 2])  # 应全部为30


x11=torch.arange(72).reshape(2,3,4,3)
vec=torch.tensor([1,2,3])
print(vec.shape)
x22=x11+vec
print(x22[0,0,0]-x11[0,0,0])
print(x22[0,1]-x11[0,1])
print(x22[0,2,0,0]-x11[0,2,0,0])
print(x22[0,2,0,2]-x11[0,2,0,2])


vec=torch.tensor([1,2,3]).reshape(1,3,1,1)
print(vec.shape)
x22=x11+vec
print(x22[0,0,0]-x11[0,0,0])
print(x22[0,1]-x11[0,1])
print(x22[0,2,0,0]-x11[0,2,0,0])
print(x22[0,1,0,0]-x11[0,1,0,0])
