import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
y_pred=[0,0,1,1,0,1]
y_true=[1,1,1,0,0,1]
target_names=['class 0','class 1']
print(classification_report(y_true,y_pred,target_names=target_names))

y_pred=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y_true=[1,1,1,0,0,2,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
target_names=['class 0','class 1','class 2']
print(classification_report(y_true,y_pred,target_names=target_names))


y_pred=[1,1,0,0,2,1,2,0,1,0,1,2,2,0,1,2,1,1,1,1]
y_true=[1,1,1,0,0,2,1,0,1,1,1,1,2,1,0,1,2,1,1,1]
print(confusion_matrix(y_true,y_pred))

#     预测  0.  1.   2
#真实.      
#0          2.  1.  1
#1          3.  7.  3
#2          0.  2.  1

# 新增：导入学习计划配套core.py中的评价函数。
import sys
sys.path.insert(
    0,
    "/Users/ersan/codex_workspace/reaserch direction/01_12周学习/12周学习配套"
)
from core import metrics

# 新增：沿用上面第五题的y_true和y_pred，自己统计混淆矩阵。
# 行是真实类别，列是预测类别，类别顺序为0、1、2。
my_matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

for i in range(len(y_true)):
    true_label = y_true[i]
    pred_label = y_pred[i]
    my_matrix[true_label][pred_label] += 1

print("自己统计的矩阵：")
for row in my_matrix:
    print(row)

# 新增：用同一组标签调用配套函数，3表示类别总数。
result = metrics(y_true, y_pred, 3)
core_matrix = result["confusion"]


print("core.py计算的矩阵:")
for row in core_matrix:
    print(row)

assert my_matrix == core_matrix, "自己统计的矩阵与core.py不一致"

# sklearn返回数组，转为列表后，就能使用同样的方式比较。
sklearn_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
assert my_matrix == sklearn_matrix.tolist(), "自己统计的矩阵与sklearn不一致"
print("核对通过:手工计数、core.py和sklearn的混淆矩阵完全一致")


print(result)