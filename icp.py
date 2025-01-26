import open3d as o3d
import numpy as np
from sklearn.neighbors import  NearestNeighbors


class ICP_Z:
    def __init__(self):
        pass
    def process(self, source, target, max_iter=2000, tolerance=1e-5):
        transformation = np.eye(4)

        for i in range(max_iter):
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
            distance, indices = nbrs.kneighbors(source)

            closest_points = target[indices.flatten()]

            source_centroid = np.mean(source, axis=0)
            closest_centroid = np.mean(closest_points, axis=0)

            source_centered = source - source_centroid
            closest_centered = closest_points - closest_centroid

            H = np.dot(source_centered.T, closest_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            t = closest_centroid - np.dot(R, source_centroid)

            # 更新变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            transformation = np.dot(T, transformation)

            # 更新源点云
            source = np.dot(R, source.T).T + t

            # 检查收敛条件
            if np.linalg.norm(t) < tolerance:
                print(f"Converged at iteration {i}")
                break
        return transformation



def read_ply(path):
    mesh = o3d.io.read_point_cloud(path)
    return mesh

def get_trans():
    angle = np.pi / 4  # 45 度
    Tmatrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0.1],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return Tmatrix

bunny_0_mesh = read_ply("data/bunny.ply")
bunny_t_mesh = read_ply("data/bunny.ply")
bunny_t_mesh.transform(get_trans())
# remove some point clouds
bunny_t_mesh.estimate_normals()
normals = np.asarray(bunny_t_mesh.normals)
mask = normals[:, 2] > 0.1
filtered_points = np.asarray(bunny_t_mesh.points)[mask]
bunny_t_mesh = o3d.geometry.PointCloud()
bunny_t_mesh.points = o3d.utility.Vector3dVector(filtered_points)


o3d.io.write_point_cloud("data/bunny_t.ply", bunny_t_mesh)

bunny_t_mesh = read_ply("data/bunny_t.ply")
bunny_t_mesh.transform(np.eye(4))


wrapper = ICP_Z()
T_z = wrapper.process(np.asarray(bunny_0_mesh.points), np.asarray(bunny_t_mesh.points))



# 进行 ICP 配准
threshold = 0.02  # 设置 ICP 的匹配阈值
trans_init = np.identity(4)  # 初始变换矩阵（单位矩阵）
reg_p2p = o3d.pipelines.registration.registration_icp(
    bunny_0_mesh, bunny_t_mesh, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

T = reg_p2p.transformation.copy()
# T[0, 3] = 0

bunny_0_mesh.transform(T)

# 可视化配准后的点云
o3d.visualization.draw_geometries([bunny_t_mesh, bunny_0_mesh], window_name="Registered Point Clouds")



# 输出配准结果
np.set_printoptions(suppress=True)

print("T_z: ")
print(T_z)

print("Transformation matrix:")
print(T)
print("#"*30)
print(get_trans())


