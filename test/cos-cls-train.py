import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from getmodel4 import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR

model = get_model(6)
loaded_dataset = torch.load('filtered_traindata.pt')
loaded_labels = torch.load('filtered_trainlabels.pt')

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device( 'cpu')

print(f'Using device: {device}')

# 创建 TensorDataset 对象
dataset = TensorDataset(loaded_dataset, loaded_labels)

# 定义批量大小
batch_size = 32 # 假设批量大小为 16

# 创建 DataLoader 对象
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = get_model(6).to(device)



    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0.001)
checkpoint_path = 'trained_model1.pth'
try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
except FileNotFoundError:
    start_epoch = 0
    print("No checkpoint found, starting from scratch.")
epoches = start_epoch + 200

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
            Accuracy = correct / total
    if Accuracy > 0.75:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/len(data_loader),
            }, f'trained_model-acc={Accuracy}.pth')
    print(f'Accuracy: {100 * correct / total}%') 
    
lr_values = []
train_losses = []

for epoch in range(start_epoch, epoches):  # 训练 5 个 epoch
    model.train()
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
    
    train_losses.append(running_loss/len(data_loader))
    lr_values.append(optimizer.param_groups[0]['lr'])
    scheduler.step(epoch + epoch/len(data_loader))
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    if (epoch+1)%5==0:
        evaluate_model(model, data_loader)
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/len(data_loader),
            }, 'trained_model1.pth')
        print("Model saved to trained_model.pth")

