import torch
import open3d as o3d
# 加载数据和标签
loaded_dataset = torch.load('testdata.pt')
loaded_labels = torch.load('testlabels.pt')


# def find_longest_distance(points):
#     num_points = points.shape[0]
#     dist_matrix = torch.cdist(points, points)  # 计算点云中所有点对之间的距离
#     dist_matrix[torch.eye(num_points, dtype=bool)] = 0  # 对角线设置为0，避免自身距离
#     max_dist, max_idx = torch.max(dist_matrix, dim=0)  # 找到最大距离及其索引
#     point1_idx = max_idx[max_dist.argmax()]
#     point2_idx = max_dist.argmax()
#     point1 = points[point1_idx]
#     point2 = points[point2_idx]
#     return point1, point2
#
#
# def rotation_matrix_from_vectors(vec1, vec2):
#     """
#     Find the rotation matrix that aligns vec1 to vec2
#     :param vec1: A 3d "source" vector
#     :param vec2: A 3d "destination" vector
#     :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
#     """
#     a = vec1 / torch.norm(vec1)
#     b = vec2 / torch.norm(vec2)
#     a1 = a.to(torch.float64)
#     b1 = b.to(torch.float64)
#     v = torch.cross(a1, b1)
#
#     c = torch.dot(a1, b1)
#
#     s = torch.norm(v)
#     kmat = torch.tensor([[0, -v[2], v[1]],
#                          [v[2], 0, -v[0]],
#                          [-v[1], v[0], 0]], dtype=torch.float64)
#     rotation_matrix = torch.eye(3) + kmat + (torch.mm(kmat, kmat)) * ((1 - c) / (s ** 2))
#     return rotation_matrix
#
#
# def align_point_cloud(points):
#     point1, point2 = find_longest_distance(points)
#     direction_vector = point2 - point1
#     target_vector = torch.tensor([0, 0, torch.norm(direction_vector)], dtype=torch.float32)
#
#     rot_matrix = rotation_matrix_from_vectors(direction_vector, target_vector)
#
#     aligned_points = torch.mm(points - point1, rot_matrix) + point1
#     return aligned_points
#
#
# def rotate_point_cloud(points, R):
#     """
#     将点云按照旋转矩阵 R 进行旋转
#     :param points: 输入的点云张量，形状为 (N, 3)
#     :param R: 旋转矩阵，形状为 (3, 3)
#     :return: 旋转后的点云张量，形状与 points 相同
#     """
#     # 将 points 转置为 (3, N)，以便与 R 相乘
#     points = points.t()
#
#     # 将旋转矩阵 R 与 points 相乘
#     rotated_points = torch.mm(R, points)
#
#     # 将结果再次转置为 (N, 3)
#     rotated_points = rotated_points.t()
#
#     return rotated_points

# import torch

def find_longest_distance(points):
    num_points = points.shape[0]
    dist_matrix = torch.cdist(points, points)  # 计算点云中所有点对之间的距离
    dist_matrix[torch.eye(num_points, dtype=bool)] = 0  # 对角线设置为0，避免自身距离
    max_dist, max_idx = torch.max(dist_matrix, dim=0)  # 找到最大距离及其索引
    point1_idx = max_idx[max_dist.argmax()]
    point2_idx = max_dist.argmax()
    point1 = points[point1_idx]
    point2 = points[point2_idx]
    return point1, point2


def calculate_surface_area(points):
    if points.shape[0] < 3:
        return 0  # 不足三个点不能构成有效的面

    # 将 PyTorch 张量转换为 NumPy 数组
    points_np = points.numpy()

    # 计算 Delaunay 三角剖分
    try:
        tri = Delaunay(points_np)
    except:
        return 0  # 如果计算失败则返回0

    # 计算每个三角形的面积
    def triangle_area(a, b, c):
        # 计算三角形的面积
        ab = b - a
        ac = c - a
        return np.linalg.norm(np.cross(ab, ac)) / 2.0

    surface_area = 0
    for simplex in tri.simplices:
        a, b, c = points_np[simplex]
        surface_area += triangle_area(a, b, c)

    return surface_area
def mark_segments(points, point1, point2):
    direction_vector = point2 - point1
    direction_vector = direction_vector / torch.norm(direction_vector)  # 归一化方向向量

    # 计算每个点到point1的投影距离
    projections = torch.matmul(points - point1, direction_vector)  # 使用矩阵乘法计算投影

    # 排序投影点
    sorted_projections, sorted_indices = torch.sort(projections)

    # 确定前1/5和后1/5的分界点
    num_points = points.shape[0]
    front_threshold_idx = num_points // 5
    back_threshold_idx = num_points * 4 // 5

    front_threshold = sorted_projections[front_threshold_idx]
    back_threshold = sorted_projections[back_threshold_idx]

    # 标记点
    labels = torch.zeros(num_points, dtype=torch.int)  # 0表示未标记部分
    front_points = labels[sorted_indices[:front_threshold_idx]]
    back_points = labels[sorted_indices[back_threshold_idx:]]
    front_points1 = points[sorted_indices[:front_threshold_idx]]
    back_points1 = points[sorted_indices[back_threshold_idx:]]
    point11, point12 = find_longest_distance(front_points1)
    point21, point22 = find_longest_distance(back_points1)
    direction_vector1 = torch.norm(point12 - point11)
    direction_vector2 = torch.norm(point22 - point21)
    if direction_vector1 < direction_vector2:
        labels[sorted_indices[:front_threshold_idx]] = 1  # 1表示尖端部分
        labels[sorted_indices[back_threshold_idx:]] = 2  # 2表示底部部分
    else:
        labels[sorted_indices[:front_threshold_idx]] = 2  # 2表示底部部分
        labels[sorted_indices[back_threshold_idx:]] = 1  # 1表示尖端部分


    return labels



# 初始化新的数据和标签列表
filtered_dataset = []
filtered_labels = []
seg_labels = []
labelall = []
for data, label in zip(loaded_dataset, loaded_labels):
    # 找到最长的两点和最大距离

    n = label*3
    # 找到最长的两点
    point1, point2 = find_longest_distance(data)

    # 标记前1/5和后1/5部分
    labels = mark_segments(data, point1, point2)
    points_np = data.numpy()
    labels = labels.numpy()
    for labelss in labels:
        seg_labels.append(n+labelss)


    # maxs, ind_maxs = torch.max(point, dim=0)
    # mins, ind_mins = torch.min(point, dim=0)
    # thred = 1 / 5 * (maxs - mins)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_np)
    # # # 将列表转换为张量
    #
    # colors = [
    #     [1, 0, 0] if labelss % 3 == 0 else [0, 0, 1] if labelss % 3 == 1 else [0, 1, 0]
    #     for labelss in labels
    # ]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd], window_name='3D Point Cloud with Labels')



# 过滤掉标签为6的数据
# for data, label in zip(loaded_dataset, loaded_labels):
#     points = []
#     if label != 6:
#         filtered_dataset.append(data)
#         filtered_labels.append(label)
#         n = label*3
#         dis_o = torch.norm(data, dim=1)
#         maxs, ind_maxs = torch.max(dis_o, dim=0)
#         vector = data[ind_maxs]
#         y_axis = torch.tensor([0.0, 1.0, 0.0]).to(torch.float64)
#         # 计算向量与 y 轴的夹角的余弦值
#         cos_theta = torch.dot(vector, y_axis) / (torch.norm(vector) * torch.norm(y_axis))
#
#         # 计算夹角的弧度
#         theta_radians = torch.acos(cos_theta)
#
#         cos_theta = torch.cos(theta_radians)
#         sin_theta = torch.sin(theta_radians)
#
#         # 构造旋转矩阵
#         R_x = torch.tensor([[1, 0, 0],
#                             [0, cos_theta, -sin_theta],
#                             [0, sin_theta, cos_theta]], dtype=torch.float64)
#
#         rotated_points = rotate_point_cloud(data, R_x)
#
#         for point in rotated_points:
#             points.append(point[1])
#
#         pointss = torch.tensor(points)
#
#         maxs, ind_maxs = torch.max(pointss, dim=0)
#         mins, ind_mins = torch.min(pointss, dim=0)
#         thred = 1 / 5 * (maxs - mins)
#
#
#         for point in points:
#             if maxs - point <= thred:
#                 seg_labels.append(n+1)
#             elif point - mins <=thred:
#                 seg_labels.append(n+2)
#             else:
#                 seg_labels.append(n)
#
#         points_np = rotated_points.numpy()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_np)
#         # # 将列表转换为张量
#
#         colors = [[1, 0, 0] if label % 3 == 0 else [0, 0, 1] for label in seg_labels]
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#         o3d.visualization.draw_geometries([pcd], window_name='3D Point Cloud with Labels')
#
#
#
# filtered_dataset = torch.stack(filtered_dataset)
# filtered_labels = torch.tensor(filtered_labels)
seg_labels = torch.tensor(seg_labels)
#
print(seg_labels)
torch.save(seg_labels,f'testseglabels.pt')