import numpy as np
import os
import torch
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import random as rd
from tqdm import tqdm

def pc_normalize(points):
    points = torch.tensor(points)
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    points = points / furthest_distance
    return points


def fps(data, n, device):
    data = torch.tensor(data).to(device)
    numpoints, d = data.shape
    centroids = torch.zeros(n, dtype=torch.long).to(device)
    distances = (torch.ones(numpoints) * 100000).type(torch.float64).to(device)
    farthest = torch.randint(0, numpoints,(1,)).to(device)
    for i in tqdm(range(n)):
        centroids[i] = farthest
        centroid = data[farthest].to(device)
        distance = torch.sum((data - centroid) ** 2, -1).to(device)
        mask = distance < distances
        distances[mask] = distance[mask]
        farthest = torch.max(distances, -1).indices.to(device)
    return centroids


class ModelDataloader(Dataset):
    def __init__(self, root, numpoints=4096):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.root = root
        self.classes = [line.rstrip() for line in open(os.path.join(self.root, 'type name.txt'))]
        self.filename = [line.rstrip() for line in open(os.path.join(self.root, 'fpstestfilenames.txt'))]
        self.num_file = [line.rstrip() for line in open(os.path.join(self.root, 'testfilenums.txt'))]
        self.filepath = []
        self.labels = []
        self.class_name = []
        self.get_filepath()
        self.numpoints = numpoints

    def __len__(self):
        return len(self.filepath)

    def get_filepath(self):
        n = -1
        for i in range(len(self.classes)):
            for j in range(int(self.num_file[i])):
                n += 1
                self.filepath.append(os.path.join(self.root, 'Typological Forms', 'fps_predict', self.classes[i], self.filename[n]))
                self.labels.append(i)
                self.class_name.append(self.classes[i])

    def __getitem__(self, idx):
        file = o3d.io.read_point_cloud(f'{self.filepath[idx]}')
        points = np.asarray(file.points)
        cla = self.labels[idx]
        centrioids = fps(points, self.numpoints, self.device).cpu()
        points = points[centrioids]
        points = pc_normalize(points)
        return points, cla


rootpath = 'D:/my code/project'
dataset = ModelDataloader(rootpath)
bs = 16
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_data = []
all_labels = []

for batch_data in dataloader:
    data_batch, labels_batch = batch_data
    all_data.append(data_batch)
    all_labels.append(labels_batch)

# 将所有批次的数据和标签连接起来
all_data = torch.cat(all_data, dim=0)
all_labels = torch.cat(all_labels, dim=0)

torch.save(all_data,f'testdata.pt')
torch.save(all_labels,f'testlabels.pt')