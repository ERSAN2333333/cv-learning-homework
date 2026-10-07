import torch
import torch.nn as nn

X=torch.rand(2,3,32,32)
conv1=nn.Conv2d(3,8,kernel_size=3,stride=2,padding=1)
X=conv1(X)
print("形状是:",X.shape)   #(2,8,16,16)
print("参数量:",conv1.weight.numel()+conv1.bias.numel())    #3*8*3*3+8=224

# W03-D06：训练片段排错练习

# 要求：三个片段分别检查。指出错误位置，说明原因和影响，写出修改代码及验证方式。
# 以下片段用于阅读排错，不是完整的可运行训练程序。

# 一、BN模式错误
# 前提：model包含BatchNorm层，使用默认的运行统计设置。下面正在进行验证。

# 待检查代码：
# model.train()
# with torch.no_grad():
#     for image, label in val_loader:
#         pred = model(image)

# 问题：
# 1. 验证时这里的模型模式是否正确？
# 2. torch.no_grad()能否阻止BN的运行均值发生变化？
# 3. 应该修改哪一行？
# 4. 如何通过比较验证前后的BN运行均值，检查修改是否有效？

# 1.不正确
# 2.若在model.train()下不可以，torch.no_grad()只能阻断梯度更新，但BN参数还是会计算更新
# 3.model.trian()改为model.eval()
# 4.可以使用model.name_buffers()方法找到BN层的参数进行对比

# 错误位置：model.train()错误
# 错误原因：这应该是模型的推理阶段，应使用model.eval()进行推理
# 可能影响：BN层参数在推理阶段也会更新
# 修改后的代码：model.trian()改为model.eval()
# 验证方式：可以使用model.name_buffers()方法找到BN层的参数进行对比

# 【批改】基本方向正确，术语和验证方法需要订正。
# 1. 验证应改为model.eval()，你判断正确。注意拼写是train，不是trian。
# 2. no_grad()关闭梯度记录，不会自动切换模型模式，也不负责清除已有梯度。
#    train模式下仍可能更新的是BN的running_mean、running_var等缓冲区，
#    不是可训练参数weight和bias；没有优化器更新时，不能说这些参数因此被更新。
# 3. 正确方法名是model.named_buffers()，不是name_buffers()。
# 4. 对比前必须clone()保存副本，否则保存的引用可能随原缓冲区一起变化。

# 【参考答案】
# 错误位置：验证前调用model.train()。
# 原因及影响：默认设置下，BN在train模式使用当前批次统计并更新运行统计，
# 验证数据会污染运行统计，预测还会受批次组成影响。no_grad()不能解决模式错误。

# 修改及验证代码（接在已有model、val_loader定义之后）：
# model.eval()
# before = {
#     name: buffer.detach().clone()
#     for name, buffer in model.named_buffers()
#     if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
# }
# assert before, "未找到BN运行统计，请检查模型"
# with torch.no_grad():
#     for image, label in val_loader:
#         image = image.to(next(model.parameters()).device)
#         pred = model(image)

# after = dict(model.named_buffers())
# for name, old_value in before.items():
#     assert torch.equal(old_value, after[name]), f"验证时BN统计发生变化：{name}"

# 说明：以上检查运行统计是否保持不变；下一轮恢复训练前，应调用model.train()。


# 二、训练集与验证集划分错误
# 前提：同一个数据集至少有2500张图片，索引唯一对应图片。希望训练集2000张、验证集500张。

# 待检查代码：
# train_indices = list(range(2000))
# val_indices = list(range(1800, 2300))

# 问题：
# 1. 两组索引是否重叠？重叠多少张？
# 2. 这种划分对验证结果有什么影响？
# 3. 在保持训练集2000张、验证集500张的条件下修改划分。
# 4. 写一个assert，检查训练和验证索引没有交集。

# 1.重叠了1800-1999索引的图片，共200张
# 2.验证结果在这些重复的图片上效果会非常好
# 3.val_indices = list(range(2000, 2500))
# 4.assert len(set(train_indices.append(val_indices)))==len(train_indices)+len(val_indices)

# 错误位置：val_indices = list(range(1800, 2300))
# 错误原因：验证集range是从1800到2299索引的图片，而训练集是从0到1999索引的图片，1800-1999的图片数据泄露了
# 可能影响：发生了数据泄露，那模型在验证阶段会对这部分数据泄露的图片表现优异，这样参考性就低了
# 修改后的代码：val_indices = list(range(2000, 2300))
# 验证方式：求这两个list的并集，若有并集则数据划分错误数据泄漏了

# 【批改】重叠数量和泄漏判断正确，修改代码与断言需要订正。
# 1. 重叠索引1800—1999，共200张，正确。
# 2. 泄漏会使验证结果可能偏乐观，不能保证重复图片一定预测得非常好。
# 3. 前面写的range(2000, 2500)正确；后面写成range(2000, 2300)只有300张，
#    不满足验证集500张的要求。
# 4. 应检查“交集”，不是“并集”。没有重叠的两组数据也有并集。
# 5. list.append()原地修改列表并返回None；还会把整个val_indices作为一个元素加入。
#    因此set(train_indices.append(val_indices))会报错，并且会破坏原训练索引列表。

# 【参考答案】
# train_indices = list(range(2000))
# val_indices = list(range(2000, 2500))

# assert len(train_indices) == 2000
# assert len(val_indices) == 500
# assert not (set(train_indices) & set(val_indices)), "训练集和验证集存在重叠"

# 说明：&求交集，空集合表示没有共同索引。这里满足题设的数量和互斥要求。
# 实际项目还要注意同一对象、相邻切片等关联样本的泄漏，仅索引不同未必足够。


# 三、平均loss计算错误
# 前提：验证集500张，batch_size=32，drop_last=False。
# loss_fn返回当前批次所有样本loss的平均值，各样本权重相同。
# 本题假定model.eval()和torch.no_grad()已正确设置，只检查均值计算。

# 待检查代码：
# total_loss = 0
# for image, label in val_loader:
#     pred = model(image)
#     loss = loss_fn(pred, label)
#     total_loss += loss.item()

# average_loss = total_loss / len(val_loader)

# 问题：
# 1. 一共有多少个批次？最后一个批次有多少张图片？
# 2. 当前代码算的是“批次均值的平均”，还是“全部样本loss的平均”？两者是否总相等？
# 3. 怎样根据每个批次的实际样本数，计算全部验证样本的平均loss？
# 4. 构造两个样本数不同、平均loss也不同的批次，用手算结果验证修改后的计算。

# 1.一共有16个批次，最后一批有20张图片
# 2.算得是批次均值的平均，两者不相等，因为最后一批并不是32张，差异就在于最后一批的个数
# 3.应该用当前批次的loss*当前批次的数量，最后total_loss在除以样本总数
# 4.

# 错误位置：total_loss += loss.item()
# 错误原因：算得批次loss直接就加到total_loss了
# 可能影响：结果会很小，而且若最后一批batch不是32，得到的结果也不准确
# 修改后的代码：total_loss += loss.item()*len(label),
# 验证方式：

# 【批改】批次数与加权思路正确，验证未完成，影响描述需要订正。
# 1. 500 = 15×32 + 20，所以共16批、尾批20张，正确。
# 2. 两种平均“不总相等”，不是“一定不相等”。例如各批loss均值相同时仍相等。
# 3. 按实际批次样本数加权是正确思路，但还要将最后的分母改为样本总数。
# 4. 原计算不一定偏小，也可能偏大：它给20张的尾批与32张的整批相同权重，
#    相当于尾批中的每张样本被赋予更高权重。
# 5. 修改代码末尾不要加逗号；在Python中它会使右侧成为元组，无法这样与数值累加。
# 6. 第4问的手算例子和验证方式尚未作答。

# 【参考答案】
# total_loss = 0.0
# total_samples = 0
# for image, label in val_loader:
#     pred = model(image)
#     loss = loss_fn(pred, label)
#     batch_size = label.size(0)
#     total_loss += loss.item() * batch_size
#     total_samples += batch_size

# assert total_samples > 0, "验证集为空"
# average_loss = total_loss / total_samples

# 说明：此处沿用题设，eval、no_grad和设备处理已在外围正确设置。
# 乘当前批次样本数，将批次平均loss还原成该批次loss总和，再除以全部样本数。

# 手算验证：
# A批2张，平均loss为1；B批1张，平均loss为4。
# 错误的按批次平均：(1 + 4) / 2 = 2.5。
# 正确的按样本平均：(1×2 + 4×1) / 3 = 2。
# 可以用逐样本loss [1, 1, 4]核对：(1 + 1 + 4) / 3 = 2。

# 对应验证代码：
# batch_sizes = [2, 1]
# batch_mean_losses = [1.0, 4.0]
# weighted_mean = sum(n * loss for n, loss in zip(batch_sizes, batch_mean_losses)) / sum(batch_sizes)
# sample_mean = sum([1.0, 1.0, 4.0]) / 3
# assert abs(weighted_mean - sample_mean) < 1e-12

# 【总体结论】
# 三类错误的核心方向均已识别。主要薄弱点是：BN参数与缓冲区的区别、
# 交集与并集及append返回值、均值计算的完整代码与验证。
# 以上参考代码为订正内容，不代表你的原作答已独立完成全部验证。

