import torch
import torch.nn as nn
import torch.nn.functional as F
from multimodal3 import PointNetSetAbstraction, PointNetSetAbstractionssg
from partseg import PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.msg1 = PointNetSetAbstraction(512, [0.1, 0.2, 0.4], [16, 32, 64], 0, [[32, 32, 64], [64, 64, 128], [128, 128, 256]])
        self.msg2 = PointNetSetAbstraction(256, [0.2, 0.4, 0.8], [32, 64, 128], 448, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.ssg = PointNetSetAbstractionssg(None, None, None, 643, [128, 256, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=1152, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=704, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=140, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, points, cls_label):
        device = points.device  # 获取输入数据所在的设备
        bs, N, _ = points.shape
        points = points.permute(0, 2, 1).to(device)
        l0_points = points
        l0_xyz = points

        l1_xyz,  l1_points = self.msg1(l0_points, l0_xyz)

        l2_xyz, l2_points = self.msg2(l1_xyz, l1_points)

        l3_xyz, l3_points = self.ssg(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(bs, 6, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points



