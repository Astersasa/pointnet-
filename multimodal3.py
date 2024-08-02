import torch
import torch.nn as nn
import torch.nn.functional as F


def fps(data, n):
    device = data.device
    bs, numpoints, d = data.shape
    idx_centroids = torch.zeros(bs, n, dtype=torch.long).to(device)
    distances = torch.ones(bs, numpoints, dtype=torch.float64).to(device) * 100000
    farthest = torch.randint(0, numpoints, (bs,), dtype=torch.long).to(device)
    batch = torch.arange(bs, dtype=torch.long).to(device)
    for i in range(n):
        idx_centroids[:, i] = farthest
        centroid = data[batch, farthest, :].view(bs, 1, 3)
        distance = torch.sum((data - centroid) ** 2, -1)
        mask = distance < distances
        distances[mask] = distance[mask]
        farthest = torch.max(distances, -1)[1]
    return idx_centroids


def Rgroup(points, centroids, radius, num_nearby):
    bs, n, c = points.shape
    _, nc, _ = centroids.shape
    # 计算每个点导中心点的距离
    # 在这里，我们希望距离形状为(B, n, N)，因此需要改变centroids和points的广播顺序
    idx = torch.arange(n, dtype=torch.long).view(1, 1, n).repeat([bs, nc, 1])
    distances = dis(points, centroids)
    # 检查截取的点是否小于半径，大于半径的赋值为无穷大
    idx[distances > radius ** 2] = 10000
    # 将距离按从小到大排序
    # distances为排序后距离的数值，idx为对应点的索引
    distances, idx = torch.sort(distances, dim=1)
    # 截取每组需要的点个数
    # idx形状为(B, n, num_nearby)
    first = idx[:, :, 0].view(bs, nc, 1).repeat([1, 1, num_nearby])
    idx = idx[:, :, :num_nearby]  #:num_nearby是前n个点，否则是第n个点
    mask = idx == 10000
    # 将标记为的点赋上第一个点的值。（？如果第一个的也是无穷大该怎么办？加个判断？）
    idx[mask] = first[mask]
    # 最终输出为索引
    return idx

    # 最大池化提取特征向量 写完发现好像不需要？


def max_pooling(merge_vector):
    # feature_vector形状为(B, n, c)
    feature_vector, _ = torch.max(merge_vector, dim=-2)
    return feature_vector


def grouping_all(points, feature):
    points = points.permute(0, 2, 1)
    new_feature = torch.cat((points, feature), dim=-2)
    new_feature = new_feature.unsqueeze(-1)
    new = points
    return new, new_feature

    # 高级索引模块


def index_points(idx, points):
    bs = points.shape[0]
    S = idx.shape[1]
    C = points.shape[2]
    centriodss = points.gather(dim=1, index=idx.unsqueeze(2).expand(bs, S, C))
    return centriodss


def group_points(idx, points):
    idx = idx
    points = points.permute(0, 2, 1)  # (b,n,c)
    bs = points.shape[0]
    n = points.shape[1]
    s = idx.shape[1]
    d = idx.shape[2]
    C = points.shape[2]
    points = points.unsqueeze(2).expand(bs, n, d, C)
    idx = idx.unsqueeze(-1).expand(bs, s, d, C)
    grouppointss = points.gather(dim=1, index=idx)
    grouppointss = grouppointss.to("cpu")
    return grouppointss


def dis(points, centriods):
    expanded_points = points.unsqueeze(1)  # 形状变为 (batch_size, 1, npoints, point_dim) 因为广播机制需要等于1，或形状相等
    expanded_centroids = centriods.unsqueeze(2)
    distances = torch.sum((expanded_points - expanded_centroids) ** 2, dim=-1)
    return distances


class PointNetSetAbstractionssg(nn.Module):
    def __init__(self, n, radius, num_nearby_values, add_attribute, mlp):
        super(PointNetSetAbstractionssg, self).__init__()
        self.n = n  # 点数
        self.radius = radius  # 半径列表
        self.num_nearby_values = num_nearby_values  # 临近点数量列表
        self.add_attribute = add_attribute  # 是否添加额外属性
        self.mlp_convs = nn.ModuleList()  # MLP卷积层列表
        self.mlp_bns = nn.ModuleList()  # MLP批量归一化层列表
        for i in range(len(mlp)):
            if i == 0:
                in_channel = add_attribute + 3
            out_channel = mlp[i]
            convs = nn.Conv2d(in_channel, out_channel, 1)
            bns = nn.BatchNorm2d(out_channel)
            self.mlp_convs.append(convs)
            self.mlp_bns.append(bns)
            in_channel = out_channel
        # 根据每层的实际输入通道数来初始化卷积层

    def forward(self, points, feature_vector):
        device = points.device
        # 获取批次大小
        # data (b,c,n)
        bs, _, _ = points.shape
        device = points.device
        # 如果不添加额外特征，则数据本身当作特征运算
        if self.add_attribute == 0:
            feature_vector = points
        else:
            feature_vector = torch.cat((points, feature_vector), dim=-2)
        points = points.permute(0, 2, 1)  # (b,n,c)
        # points形状变为(B, c N)
        # feature形状为(B, c, N)或(B, c, N+n)
        # 初始化总体特征张量
        total_feature = None
        new_points, new_feature = grouping_all(points, feature_vector)
        new_feature = new_feature.to(torch.float32).to(device)
        for i in range(len(self.mlp_convs)):
            convs = self.mlp_convs[i].to(device)
            bn = self.mlp_bns[i].to(device)
            new_feature = F.relu(bn(convs(new_feature)))
        total_feature = torch.max(new_feature, 2)[0]
        new_points = new_points.permute(0, 2, 1)  # (b,c,n)
        total_feature = total_feature.to(device)
        return new_points, total_feature


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n, radius, num_nearby_values, add_attribute, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.n = n  # 点数
        self.radius = radius  # 半径列表
        self.num_nearby_values = num_nearby_values  # 临近点数量列表
        self.add_attribute = add_attribute  # 是否添加额外属性
        self.mlp_convs = nn.ModuleList()  # MLP卷积层列表
        self.mlp_bns = nn.ModuleList()  # MLP批量归一化层列表
        for i in range(len(mlp)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            in_channel = add_attribute + 3
            for out_channel in mlp[i]:
                convs.append(nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1))
                bns.append(nn.BatchNorm2d(out_channel))
                in_channel = out_channel
            self.mlp_convs.append(convs)
            self.mlp_bns.append(bns)
        self.mlp = mlp
        # 根据每层的实际输入通道数来初始化卷积层

    # points是所有点云数据，形状为(B, c, N)
    # feature_vector是特征，形状为(B, c, N)
    def forward(self, points, feature_vector):
        # 获取批次大小
        # data (b,c,n)
        bs, _, _ = points.shape
        device = points.device
        # 如果不添加额外特征，则数据本身当作特征运算
        if self.add_attribute == 0:
            feature_vector = points
        else:
            feature_vector = torch.cat((points, feature_vector), dim=-2)
        points = points.permute(0, 2, 1)  # (b,n,c)
        # points形状变为(B, c N)
        # feature形状为(B, c, N)或(B, c, N+n)
        # 初始化总体特征张量
        total_feature = None
        idx_centroids = fps(points, self.n)
        centroids = index_points(idx_centroids, points)
        # 判断是否对所有点分组
        for i in range(len(self.mlp)):
            group_idx = Rgroup(points, centroids, self.radius[i], self.num_nearby_values[i])
            new_feature = group_points(group_idx, feature_vector)
            new_points = centroids
            new_feature = new_feature.permute(0, 3, 2, 1)
            new_feature = new_feature.float().to(device)
            for j in range(len(self.mlp[i])):
                # 逐层卷积和批量归一化
                convs = self.mlp_convs[i][j]
                bn = self.mlp_bns[i][j]
                new_feature = F.relu(bn(convs(new_feature)))
            # 最大池化得到特征
            new_feature = max_pooling(new_feature)
            new_feature = new_feature.double().to('cpu')
            if total_feature == None:
                total_feature = torch.empty(bs, 0, new_feature.shape[-1], dtype=torch.float64)
            total_feature = torch.cat((total_feature, new_feature), dim=-2)
        new_points = new_points.permute(0, 2, 1)  # (b,c,n)
        total_feature = total_feature.to(device)
        return new_points, total_feature

    # SetAbstraction第一步：最远端采样
