import pathlib

from torch.utils.data import Dataset
import glob
import os
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
# from sklearn.neighbors import NearestNeighbors
def real3d_classes():
    return ['airplane','car','candybar','chicken',
            'diamond','duck','fish','gemstone',
            'seahorse','shell','starfish','toffees']
def generate_random_points_in_sphere(center, radius, num_points):
    r = np.random.uniform(0, radius, num_points) ** (1/3)
    
    theta = np.random.uniform(0, np.pi, num_points)
    
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    points = np.stack((x, y, z), axis=-1) + center
    
    return points

def random_rotation_matrix():
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return centroids,point


class Dataset3dad_ShapeNet_test(Dataset):
    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.if_norm = if_norm
        test_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'test')) + '/*.pcd')
        test_sample_list = [s for s in test_sample_list if 'temp' not in s]
        cut_list = [s for s in test_sample_list if 'cut' in s or 'copy' in s]
        self.test_sample_list = test_sample_list
        self.gt_path = str(os.path.join(dataset_dir, cls_name, 'GT'))

    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points

    def __getitem__(self, idx):
        sample_path = self.test_sample_list[idx]
        if 'positive' in sample_path:
            pcd = o3d.io.read_point_cloud(sample_path)
            pointcloud = np.array(pcd.points)
            mask = np.zeros((pointcloud.shape[0]))
            label = 0
            
        else:
            filename = pathlib.Path(sample_path).stem
            txt_path = os.path.join(self.gt_path, filename + '.txt')
            pcd = np.genfromtxt(txt_path, delimiter=",")
            pointcloud = pcd[:, :3]
            mask = pcd[:, 3]
            label = 1
        
        if(self.if_norm):
            pointcloud = self.norm_pcd(pointcloud)
            
        pointcloud = pointcloud * 5
        return pointcloud, mask, label, sample_path
        #return pointcloud, mask



    def __len__(self):
        return len(self.test_sample_list)
       
      

def perturb_points_on_surface(points, move_distance=0.1, sphere=True, direction="outward"):
    perturbed_points = points.copy()

    normals = compute_normals(points)

    for i, point in enumerate(points):
        normal = normals[i]
        move_dir = normal if direction == "outward" else -normal

        move_magnitude = np.random.uniform(0, move_distance)
        new_point = point + move_dir * move_magnitude

        if sphere:
            perturbed_points[i] = new_point / np.linalg.norm(new_point) * np.linalg.norm(point)
        else:
            axis_lengths = np.random.uniform(0.8, 1.2, size=3)  # 椭球比例因子
            perturbed_points[i] = new_point * axis_lengths

    return perturbed_points

def compute_normals(points, k_neighbors=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
    normals = np.asarray(pcd.normals)
    return normals

def compute_local_normals(points, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = []
    for idx in range(points.shape[0]):
        neighbors = points[indices[idx]]
        cov_matrix = np.cov(neighbors - np.mean(neighbors, axis=0), rowvar=False)
        _, _, eigenvectors = np.linalg.svd(cov_matrix)
        normals.append(eigenvectors[-1])  # 最小特征值对应的法向量

    return np.array(normals)

def select_smooth_outer_point(pointcloud, max_angle_diff=15):
    hull = ConvexHull(pointcloud)
    convex_hull_points = pointcloud[hull.vertices]

    normals = compute_local_normals(convex_hull_points)

    angle_diffs = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cos_theta = np.dot(normals[i], normals[j]) / (
                np.linalg.norm(normals[i]) * np.linalg.norm(normals[j])
            )
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
            angle_diffs.append((i, j, angle))

    valid_indices = []
    for i, j, angle in angle_diffs:
        if angle < max_angle_diff:
            valid_indices.extend([i, j])

    valid_indices = list(set(valid_indices))
    if not valid_indices:
        raise ValueError("No suitable points found with the given constraints.")

    selected_index = np.random.choice(valid_indices)
    return convex_hull_points[selected_index]

def select_outer_point(pointcloud):
    centroid = np.mean(pointcloud, axis=0)

    hull = ConvexHull(pointcloud)
    convex_hull_points = pointcloud[hull.vertices]
    outer_point = convex_hull_points[np.random.choice(len(convex_hull_points))]

    return outer_point

def select_outer_point_with_index(pointcloud):
    centroid = np.mean(pointcloud, axis=0)

    hull = ConvexHull(pointcloud)
    convex_hull_points = pointcloud[hull.vertices]
    random_index_in_hull = np.random.choice(len(convex_hull_points))
    outer_point = convex_hull_points[random_index_in_hull]

    original_index = hull.vertices[random_index_in_hull]

    return outer_point, original_index


def apply_smooth_function(relative_distances, move_distance, smooth_fn):
    if smooth_fn == "gaussian":
        smooth_ratios = np.exp(-relative_distances**2)
    elif smooth_fn == "linear":
        smooth_ratios = 1 - relative_distances
    elif smooth_fn == "inverse":
        smooth_ratios = 1 / (1 + 10 * relative_distances)
    elif smooth_fn == "sine":
        smooth_ratios = np.sin((1 - relative_distances) * np.pi / 2)
    elif smooth_fn == "polynomial":
        smooth_ratios = 1 - relative_distances**2
    else:
        raise ValueError(f"Unsupported smooth function: {smooth_fn}")

    return smooth_ratios * move_distance

def generate_perturbed_points(points, move_distance=0.1, sphere=True):
    mask = np.zeros(points.shape[0])  # 初始化 mask，默认所有点未移动
    initial_point = select_outer_point(points)

    distances = np.linalg.norm(points - initial_point, axis=1)

    k = int(points.shape[0] * 0.02)
    nearest_indices = np.argsort(distances)[:k]
    selected_points = points[nearest_indices]

    random_value_np = np.random.rand()
    if random_value_np < 0.5:
        direction = "outward" if random_value_np < 0.25 else "inward"
        perturbed_points = perturb_points_on_surface(
            selected_points, move_distance=move_distance, sphere=sphere, direction=direction
        )
        points[nearest_indices] = perturbed_points
        mask[nearest_indices] = 1

    return points, mask, nearest_indices

def generate_smooth_perturbed_points_v2(pointcloud, move_distance=0.1, sphere=True, smooth_fn="gaussian", random_value_np=0.1):
    mask = np.zeros(pointcloud.shape[0])  # 初始化 mask

    initial_point = select_outer_point(pointcloud)

    distances = np.linalg.norm(pointcloud - initial_point, axis=1)

    k = int(pointcloud.shape[0] * 0.02)
    neighbor_indices = np.argsort(distances)[:k]
    selected_points = pointcloud[neighbor_indices]

    mask[neighbor_indices] = 1

    normals = compute_normals(selected_points)

    direction = "outward" if random_value_np < 0.25 else "inward"

    max_distance = distances[neighbor_indices[-1]]  # 最大邻近距离
    relative_distances = distances[neighbor_indices] / max_distance
    smooth_move_distances = apply_smooth_function(relative_distances, move_distance, smooth_fn)
    if direction=='inward':
        smooth_move_distances = -np.abs(smooth_move_distances)
    else:
        smooth_move_distances = np.abs(smooth_move_distances)

    deformed_points = []
    for i, point in enumerate(selected_points):
        normal = normals[i]
        move_dir = np.abs(normal)
        new_point = point + move_dir * smooth_move_distances[i]
        deformed_points.append(new_point)

    deformed_points = np.array(deformed_points)

    pointcloud[neighbor_indices] = deformed_points

    return pointcloud, mask, neighbor_indices

def generate_hole_perturbation(pointcloud):
    mask = np.zeros(pointcloud.shape[0])

    initial_point = select_outer_point(pointcloud)

    distances = np.linalg.norm(pointcloud - initial_point, axis=1)

    k_outer = int(pointcloud.shape[0] * 0.005)
    neighbor_indices = np.argsort(distances)[:k_outer]

    mask[neighbor_indices] = 1

    k_inner = int(pointcloud.shape[0] * 0.003)
    inner_indices = np.argsort(distances)[:k_inner]  # 更接近的点索引

    flag = np.ones(len(pointcloud), dtype=bool)
    flag[inner_indices] = False
    updated_pointcloud = pointcloud[flag]
    mask = mask[flag]

    neighbor_indices_new = np.where(mask == 1)[0]

    return updated_pointcloud, mask, neighbor_indices_new

def generate_scratch_perturbation(pointcloud, normals, perturb_fraction=0.05, strip_width=0.05, max_displacement=0.02):
    selected_point = select_outer_point(pointcloud)

    kdtree = cKDTree(pointcloud)
    neighbor_indices = kdtree.query_ball_point(selected_point, r=strip_width)

    neighbor_points = pointcloud[neighbor_indices]
    pca = PCA(n_components=2)
    pca.fit(neighbor_points)
    principal_direction = pca.components_[0]  # 主方向向量

    projections = np.dot(neighbor_points - selected_point, principal_direction)
    sorted_indices = np.argsort(np.abs(projections))
    num_points = int(pointcloud.shape[0] * perturb_fraction)
    strip_indices = np.array(neighbor_indices)[sorted_indices[:num_points]]

    mask = np.zeros(pointcloud.shape[0], dtype=int)
    mask[strip_indices] = 1

    selected_points = pointcloud[strip_indices]

    normals = compute_normals(selected_points)

    perturbed_pointcloud = pointcloud.copy()
    displacement = np.random.uniform(0, max_displacement, size=len(strip_indices))

    for i, idx in enumerate(strip_indices):
        perturbed_pointcloud[idx] = perturbed_pointcloud[idx] - np.abs(normals[i] * displacement[i])


    return perturbed_pointcloud, mask, strip_indices

def select_curve_strip(pointcloud, center_index, radius, length_ratio, width_ratio):
    kdtree = KDTree(pointcloud)
    center_point = pointcloud[center_index]
    neighbor_indices = kdtree.query_radius(center_point.reshape(1, -1), r=radius)
    neighbor_indices = neighbor_indices[0]

    target_point_count = int(pointcloud.shape[0] * 0.01)
    num_points = len(neighbor_indices)
    while num_points < target_point_count:
        radius *= 1.1  # 扩大半径
        neighbor_indices = kdtree.query_radius(center_point.reshape(1, -1), r=radius)[0]
        num_points = len(neighbor_indices)
    
    while num_points > target_point_count:
        radius *= 0.9  # 缩小半径
        neighbor_indices = kdtree.query_radius(center_point.reshape(1, -1), r=radius)[0]
        num_points = len(neighbor_indices)

    neighbor_points = pointcloud[neighbor_indices]
    local_normals = compute_normals(neighbor_points)  # 计算邻域法向量

    normal = local_normals.mean(axis=0)  # 近似法向量（局部平均）
    normal = normal / np.linalg.norm(normal)  # 归一化

    tangent1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    tangent1 = tangent1.astype(np.float64)
    tangent1 -= np.dot(tangent1, normal) * normal  # 垂直法向量的切向方向
    tangent1 = tangent1 / np.linalg.norm(tangent1)

    tangent2 = np.cross(normal, tangent1)  # 另一个正交方向

    relative_positions = neighbor_points - center_point
    proj_x = np.dot(relative_positions, tangent1)  # 在第一个切向方向的投影
    proj_y = np.dot(relative_positions, tangent2)  # 在第二个切向方向的投影

    length_threshold = radius * length_ratio
    width_threshold = radius * width_ratio
    strip_mask = (abs(proj_x) <= length_threshold) & (abs(proj_y) <= width_threshold)

    strip_indices = np.array(neighbor_indices)[strip_mask]  # 转换为全局索引

    return strip_indices


def generate_scratch_with_rectangular_box(pointcloud, normals, max_displacement=0.1):
    
    mask = np.zeros(pointcloud.shape[0])  # 初始化 mask

    initial_point, center_index = select_outer_point_with_index(pointcloud)

    radius = 2.0             # 邻域搜索半径
    length_ratio = 0.8       # 条带长度比例
    width_ratio = 0.2        # 条带宽度比例

    strip_indices = select_curve_strip(pointcloud, center_index, radius, length_ratio, width_ratio)
    strip_points = pointcloud[strip_indices]

    normals = compute_normals(strip_points)   # 计算这些点的法向量，有替换掉默认值

    displacement = np.random.uniform(0, max_displacement, size=len(strip_points))  # 随机生成移动的距离

    perturbed_pointcloud = pointcloud.copy()
    for i, idx in enumerate(strip_indices):
        perturbed_pointcloud[idx] = perturbed_pointcloud[idx] + np.abs(normals[i]) * displacement[i]  # 沿法向量方向移动点

    mask = np.zeros(len(pointcloud))
    mask[strip_indices] = 1  # 将条带上的点标记为1，表示它们被移动了

    return perturbed_pointcloud, mask, strip_indices

class Dataset3dad_ShapeNet_train_final_newaug(Dataset):
    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.train_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'train')) + '/*template*.pcd')
        self.if_norm = if_norm
        self.cls = cls_name

    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
    def create_pseudo_path(self,idx):
        return self.train_sample_list[(idx+1) % 4]
        
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.train_sample_list[idx])
        pointcloud = np.array(pcd.points)
        R = random_rotation_matrix()
        pointcloud = np.dot(pointcloud, R.T)

        if(self.if_norm):
            pointcloud = self.norm_pcd(pointcloud)

        
        random_value_np = np.random.rand()
        move_distance = np.random.uniform(0.005, 0.05)

        if (random_value_np < 0.5):
            updated_pointcloud, mask, neighbor_indices = generate_smooth_perturbed_points_v2(
                pointcloud, move_distance=move_distance, sphere=True, smooth_fn="sine", random_value_np=random_value_np
            )
            
        elif (random_value_np < 0.8):
            if (random_value_np < 0.65):
                updated_pointcloud, mask, neighbor_indices = generate_hole_perturbation(pointcloud)

            else:
                normals = np.random.rand(1000, 3) - 0.5
                max_displacement = 0.002

                updated_pointcloud, mask, neighbor_indices = generate_scratch_with_rectangular_box(
                    pointcloud, normals, max_displacement
                )


        else:
            mask = np.zeros((pointcloud.shape[0]))
            updated_pointcloud = pointcloud.copy()


        if self.num_points > 0:
            slice=np.random.choice(updated_pointcloud.shape[0],17422)
            updated_pointcloud = updated_pointcloud[slice]
            mask = mask[slice]


        updated_pointcloud = updated_pointcloud * 5
        label = 0
        return updated_pointcloud, mask, label, self.train_sample_list[idx]

    def __len__(self):
       return len(self.train_sample_list)


class Dataset3dad_ShapeNet_train_memory(Dataset):
    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.train_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'train')) + '/*template*.pcd')
        self.if_norm = if_norm
        self.cls = cls_name

    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
    def create_pseudo_path(self,idx):
        return self.train_sample_list[(idx+1) % 4]
        
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.train_sample_list[idx])
        pointcloud = np.array(pcd.points)
        if (self.if_norm):
            pointcloud = self.norm_pcd(pointcloud)
            mask = np.zeros((pointcloud.shape[0]))
            label = 0
        if self.num_points > 0:
            slice=np.random.choice(pointcloud.shape[0],17422)
            pointcloud = pointcloud[slice]
            mask = mask[slice]
        pointcloud = pointcloud * 5
        label = 0
        return pointcloud, mask, label, self.train_sample_list[idx]

    def __len__(self):
       return len(self.train_sample_list)
