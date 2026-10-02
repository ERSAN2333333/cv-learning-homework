#输入是(32,32),卷积核大小是(3,3),padding=1,stride=1,输出是（(32-3+2*1)/1+1）*（(32-3+2*1)/1+1）=32*32
#输入是(32,32),卷积核大小是(3,3),padding=1,stride=2,输出是（(32-3+2*1)/2+1）*（(32-3+2*1)/2+1）=16*16
#输出尺寸 = 向下取整[(输入尺寸 + 2×padding - 卷积核大小) /stride] + 1
import torch
import torch.nn as nn

X=torch.rand(3,32,32)
print(X.shape)
conv1=nn.Conv2d(3,1,kernel_size=3,padding=1,stride=1)
X=conv1(X)
print(X.shape)
conv2=nn.Conv2d(1,3,kernel_size=3,padding=1,stride=2)
X=conv2(X)
print(X.shape)

#conv1 有 1×3×3×3=27 个权重、1个偏置，共 28；conv2 有 3×1×3×3=27 个权重、3个偏置，共 30
print("conv1参数量：", conv1.weight.numel() + conv1.bias.numel())  # 28
print("conv2参数量：", conv2.weight.numel() + conv2.bias.numel())  # 30

X = torch.zeros(3, 32, 32)
X[0, 16, 16] = 1  # 第0通道，位置(16,16)放一个亮点

#“单点脉冲”就是把输入图像全部设为0，只让一个像素等于1，看这个点会影响输出的哪些位置。
conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1, bias=False)
with torch.no_grad():
    conv1.weight.fill_(1)

Y = conv1(X)
print(torch.nonzero(Y[0]))  # 查看输出哪些位置不为0

# 比较增加输入通道与增加分辨率：每次只改一个条件。
print("\n比较输入通道与分辨率")

# 基准：3通道、32×32。后面的分辨率实验会复用同一个卷积层。
image_3_32 = torch.rand(3, 32, 32)
conv_3 = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1)
output_3_32 = conv_3(image_3_32)
print("3通道、32×32：输出", output_3_32.shape,
      "参数量", conv_3.weight.numel() + conv_3.bias.numel(),
      "输出元素数", output_3_32.numel())

# 只增加输入通道：3通道变6通道；空间大小仍是32×32。
image_6_32 = torch.rand(6, 32, 32)
conv_6 = nn.Conv2d(6, 1, kernel_size=3, padding=1, stride=1)
output_6_32 = conv_6(image_6_32)
print("6通道、32×32：输出", output_6_32.shape,
      "参数量", conv_6.weight.numel() + conv_6.bias.numel(),
      "输出元素数", output_6_32.numel())

# 只增加分辨率：通道仍是3，使用与基准完全相同的conv_3。
image_3_64 = torch.rand(3, 64, 64)
output_3_64 = conv_3(image_3_64)
print("3通道、64×64：输出", output_3_64.shape,
      "参数量", conv_3.weight.numel() + conv_3.bias.numel(),
      "输出元素数", output_3_64.numel())

print("结论：通道增加使参数量28→55；分辨率增加使输出元素数1024→4096，参数量仍是28。")

Y=torch.rand(2,3,32,32)
Z=torch.rand(2,103,32,32)
conv_test=nn.Conv2d(3,8,kernel_size=3,padding=1,stride=1)
Y=conv_test(Y)
assert Y.shape==(2,8,32,32)
conv_test2=nn.Conv2d(8,9,kernel_size=3,padding=1,stride=2)
Y=conv_test2(Y)
assert Y.shape==(2,9,16,16)
assert conv_test.weight.numel()+conv_test.bias.numel()==224
assert conv_test2.weight.numel()+conv_test2.bias.numel()==657