import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from getmodel3 import get_model

loaded_dataset = torch.load('traindata.pt')
loaded_labels = torch.load('trainlabels.pt')

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建 TensorDataset 对象
dataset = TensorDataset(loaded_dataset, loaded_labels)

# 定义批量大小
batch_size = 8 # 假设批量大小为 16

# 创建 DataLoader 对象
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = get_model(7).to(device)

epoches = 1
def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    model.eval()  # 切换到评估模式

    with torch.no_grad():
        for inputs, labels in dataloader:
            # 将输入和标签迁移到GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Accuracy: {100 * correct / total}%')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器，学习率为0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(epoches):  # 训练 5 个 epoch
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc=f'Training {epoch+1}/{epoches}'):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, l3_points = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 清理缓存
        torch.cuda.empty_cache()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}")
    evaluate_model(model, data_loader)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'trained_model.pth')
    print("Model saved to trained_model.pth")








