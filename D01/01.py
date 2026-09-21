import torch
import sys

print("Python路径：", sys.executable)
print("PyTorch版本：", torch.__version__)
print("CUDA可用：", torch.cuda.is_available())

x=torch.arange(24).reshape(2,3,4)
print("第一组：",x[0])
print("第一组形状：",x[0].shape)

rgb=torch.rand(8,3,32,32)   #(B,C,H,W)
cube=torch.rand(30,30,5)   #(H,W,C)

cube_change=cube.permute(2,0,1).unsqueeze(0)
print(cube_change.shape)

print(x[1,2,3])
print(x[0].float().mean())   #第一组均值
print(x[1].float().mean())   #第二组均值

x=torch.rand(4,1,6,6)
mean_value=x.mean(dim=(1,2,3))
max_value=x.amax(dim=(1,2,3))
print("每张图均值：",mean_value)
print("每张图最大值",max_value)

torch.save(
    {
    "images":x,
    "mean":mean_value,
    "max":max_value
    },
    "day01_output.pt"
)
