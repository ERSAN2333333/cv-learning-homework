import torch
import torch.nn as nn
from pathlib import Path

class ResicBlock(nn.Module):
    def __init__(self,input_channel,num_channel,use_1x1conv=False,strides=1,drop_rate=0.3,**kwargs):
        super().__init__(**kwargs)
        self.conv1=nn.Conv2d(input_channel,num_channel,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channel,num_channel,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv1x1=nn.Conv2d(input_channel,num_channel,kernel_size=1,stride=strides)
        else:
            self.conv1x1=None
        self.bn1=nn.BatchNorm2d(num_channel)
        self.bn2=nn.BatchNorm2d(num_channel)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(drop_rate)
    def forward(self,X):
        x=X
        X=self.relu(self.bn1(self.conv1(X)))
        X=self.bn2(self.conv2(X))
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        return self.relu(self.dropout(x+X))
    
if __name__=="__main__":
    X = torch.rand(2, 16, 32, 32)
    # 1. 同形状残差块
    same_block = ResicBlock(16, 16, use_1x1conv=False, strides=1)
    print("同形状输出：", same_block(X).shape)

    # 2. 只改变通道，保持高宽不变
    channel_block = ResicBlock(16, 20, use_1x1conv=True, strides=1, drop_rate=0.4)
    print("通道变化输出：", channel_block(X).shape)

    # 3. 直接测试模型里的Dropout，固定输入为全1
    test_input = torch.ones(2, 8)
    channel_block.train()
    train_output1=channel_block.dropout(test_input)
    train_output2=channel_block.dropout(test_input)
    print("训练第一次：", train_output1)
    print("训练第二次：", train_output2)

    channel_block.eval()
    eval_output1=channel_block.dropout(test_input)
    eval_output2=channel_block.dropout(test_input)
    print("测试第一次：", eval_output1)
    print("测试第二次：", eval_output2)
    
     # 4. 检查训练模式下BN的运行均值
    channel_block.train()

    old_mean1 = channel_block.bn1.running_mean.clone()
    old_mean2 = channel_block.bn2.running_mean.clone()

    with torch.no_grad():
        channel_block(X)

    print("训练模式BN1均值变化：",
          not torch.equal(old_mean1, channel_block.bn1.running_mean))
    print("训练模式BN2均值变化：",
          not torch.equal(old_mean2, channel_block.bn2.running_mean))

    # 5. 检查评价模式下BN的运行均值
    channel_block.eval()

    old_mean1 = channel_block.bn1.running_mean.clone()
    old_mean2 = channel_block.bn2.running_mean.clone()

    with torch.no_grad():
        channel_block(X)

    print("评价模式BN1均值变化：",
          not torch.equal(old_mean1, channel_block.bn1.running_mean))
    print("评价模式BN2均值变化：",
          not torch.equal(old_mean2, channel_block.bn2.running_mean))
    
    # 6. 冻结参数，但保持BN处于训练模式
    for parameter in channel_block.parameters():
        parameter.requires_grad=False
    channel_block.train()

    old_weight=channel_block.conv1.weight.detach().clone()
    old_mean1=channel_block.bn1.running_mean.clone()
    old_mean2=channel_block.bn2.running_mean.clone()

    channel_block(X)
    print("冻结参数第一层权重变化:",not torch.equal(old_weight,channel_block.conv1.weight))
    print("冻结参数BN1运行均值变化：",
          not torch.equal(old_mean1, channel_block.bn1.running_mean))
    print("冻结参数BN2运行均值变化：",
          not torch.equal(old_mean2, channel_block.bn2.running_mean))

    # 7. 保存重复前向差异和BN buffer的实际前后值
    torch.manual_seed(42)
    Y = torch.ones(2, 3, 4, 4)
    model = ResicBlock(3, 4, use_1x1conv=True, strides=1, drop_rate=0.4)

    # named_buffers()提供名称和张量；clone()保存独立的数值快照。
    train_before = {}
    for name, buffer in model.named_buffers():
        train_before[name] = buffer.clone()

    model.train()
    with torch.no_grad():
        test1 = model(Y)
        test2 = model(Y)
    train_diff = (test1 - test2).abs().max().item()

    train_after = {}
    for name, buffer in model.named_buffers():
        train_after[name] = buffer.clone()

    # 训练结束的状态，也就是这次评价开始前的状态。
    model.eval()
    with torch.no_grad():
        test3 = model(Y)
        test4 = model(Y)
    eval_diff = (test3 - test4).abs().max().item()

    eval_after = {}
    for name, buffer in model.named_buffers():
        eval_after[name] = buffer.clone()

    log_lines = [
        "固定种子42；同一个全1输入，形状[2,3,4,4]；Dropout概率0.4。",
        "整个残差块分别在train和eval下前向两次，没有反传或更新权重。",
        f"训练两次输出是否相同：{torch.equal(test1, test2)}",
        f"训练两次最大绝对差：{train_diff}",
        f"评价两次输出是否相同：{torch.equal(test3, test4)}",
        f"评价两次最大绝对差：{eval_diff}",
    ]
    for name in train_before:
        log_lines.append(f"\n{name}")
        log_lines.append(f"训练前：{train_before[name]}")
        log_lines.append(f"训练后：{train_after[name]}")
        log_lines.append(f"评价前：{train_after[name]}")
        log_lines.append(f"评价后：{eval_after[name]}")

    log_path = Path(__file__).resolve().parent / "bn_dropout_log.txt"
    with open(log_path, "w", encoding="utf-8") as file:
        for line in log_lines:
            print(line)
            file.write(line + "\n")
    print("日志已保存到：", log_path)

    #8.补练模型中只有一个子模块保持train的情况，遍历named_modules定位。
    
    model.eval()
    print("模型整体是否为train：", model.training)
    #这里会输出 False，但不代表每个子模块都是eval。
    model.bn1.train()
    for name,module in model.named_modules():
        if module.training:
            print("仍处于训练模式的模块：",name)
    model.eval()
    for name, module in model.named_modules():
        if module.training:
            print("仍处于train的模块：", name)
