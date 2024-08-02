import torch.nn as nn
import torch.nn.functional as F
from multimodal3 import PointNetSetAbstraction, PointNetSetAbstractionssg


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.msg1 = PointNetSetAbstraction(512, [0.1, 0.2, 0.4], [16, 32, 64], 0, [[32, 32, 64], [64, 64, 128], [128, 128, 256]])
        self.msg2 = PointNetSetAbstraction(256, [0.2, 0.4, 0.8], [32, 64, 128], 448, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.ssg = PointNetSetAbstractionssg(None, None, None, 643, [128, 256, 512])
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.drop3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(128, num_class)

    def forward(self, points):
        device = points.device  # 获取输入数据所在的设备
        bs, _, _ = points.shape
        points = points.permute(0, 2, 1).to(device)
        l1_xyz,  l1_points = self.msg1(points, None)

        l2_xyz, l2_points = self.msg2(l1_xyz, l1_points)

        l3_xyz, l3_points = self.ssg(l2_xyz, l2_points)

        x = l3_points.view(bs, 512).to(device)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        x = F.log_softmax(x, -1)

        return x, l3_points

