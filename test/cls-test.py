import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from getmodel3 import get_model
from sklearn.metrics import accuracy_score, classification_report

#加载测试数据和标签
test_dataset = torch.load('testdata.pt')
test_labels = torch.load('testlabels.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')

#创建TensorDataset对象
dataset = TensorDataset(test_dataset, test_labels)

#定义批量大小
batch_size = 16

#创建DataLoader对象
#测试集一般不需要打乱数据
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_model(model_path, num_classes=6, normal_channel=False, device='cuda'):
    """
    加载 .pth 文件中的模型权重并返回模型实例

    Args:
        model_path (str): 模型权重文件的路径 (.pth 文件)
        num_classes (int): 模型的类别数量
        normal_channel (bool): 是否使用 normal_channel 参数
        device (str): 设备 ('cuda' 或 'cpu')

    Returns:
        torch.nn.Module: 加载了权重的模型实例
    """
    model = get_model(num_class=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

model_path = 'trained_model1acc=0.75 new.pth'
model = load_model(model_path, num_classes=6, device=device)

model.eval()


all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        output,_ = model(data)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 使用 sklearn 计算准确率和分类报告
instance_acc = accuracy_score(all_targets, all_preds)
class_labels = ['D', 'E', 'F', 'G', 'J', 'K']
class_report = classification_report(all_targets, all_preds, target_names=class_labels, digits=2)

print('Test Instance Accuracy: {:.2f}'.format(instance_acc))
print('Classification Report:\n{}'.format(class_report))