import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from getmodel4 import get_model

loaded_dataset = torch.load('filtered_traindata.pt')
loaded_labels = torch.load('filtered_trainlabels.pt')

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建 TensorDataset 对象
dataset = TensorDataset(loaded_dataset, loaded_labels)

# 定义批量大小
batch_size = 16 # 假设批量大小为 16

# 创建 DataLoader 对象
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = get_model(6).to(device)

epochs = 400
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器，学习率为0.001

checkpoint_path = 'trained_model.pth'
start_epoch = 0
try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch.")
def evaluate_model(model, dataloader,device):
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
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy
for epoch in range(start_epoch, epochs):
    running_loss = 0.0
    model.train()
    for inputs, labels in tqdm(data_loader, desc=f'Training {epoch+1}/{epochs}'):

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
    if (epoch + 1) % 10 == 0:
        accuracy = evaluate_model(model, data_loader, device)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(data_loader),
        }, f'trained_model_epoch_{epoch + 1}.pth')
        print(f"Model saved to trained_model_epoch_{epoch + 1}.pth")


torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(data_loader),
            }, 'trained_model.pth')
print("Model saved to trained_model.pth")








