#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   transforms.py
@Time    :   2023/05/05 12:41:00
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.eddu.com
'''
import torch
import numpy as np
import math
from torch_sparse import coalesce
from utils.helpers import compute_bond_angles


class GetY(object):
    '''Specify target for prediction'''
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]  # data.y: (#crystals, #targets)
        return data


class GetAngle(object):
    '''Computes bond angles in the crystall'''
    def __call__(self, data):
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        data.angle_index  = torch.stack([idx_kj, idx_ji], dim=0)
        data.angle_attr   = angles.reshape(-1, 1)
        return data


# ==================== 新增：GetPeriodicGeometry类 ==================== #
class GetPeriodicGeometry(object):
    '''Computes bond angles and dihedral angles in the crystal using PerCNet method'''
    
    def __call__(self, data):
        # 调用新封装的计算函数
        (
            bond_angles, 
            dihedral_phi, 
            dihedral_tau, 
            angle_index,
            dihedral_index
        ) = self.compute_periodic_geometry(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        
        # 将新的特征存入data对象
        data.angle_attr = bond_angles.reshape(-1, 1)
        data.angle_index = angle_index
        data.dihedral_phi = dihedral_phi.reshape(-1, 1)  # 第一种二面角
        data.dihedral_tau = dihedral_tau.reshape(-1, 1)  # 第二种二面角
        data.dihedral_index = dihedral_index  # 二面角的边索引
        
        return data
    
    def compute_periodic_geometry(self, pos, cell_offsets, edge_index, num_nodes):
        """
        从PerCNet的get_new_3tuple函数移植并适配的几何计算逻辑
        
        Args:
            pos: 原子坐标 [N, 3]
            cell_offsets: 晶胞偏移 [E, 3]
            edge_index: 边索引 [2, E]
            num_nodes: 节点数量
            
        Returns:
            bond_angles: 键角 [num_triplets]
            dihedral_phi: 第一种二面角 [num_4tuples]
            dihedral_tau: 第二种二面角 [num_4tuples]
            angle_index: 键角的边索引 [2, num_triplets]
            dihedral_index: 二面角的边索引 [2, num_4tuples]
        """
        
        # 1. 构建邻居关系，适配GCPNet的数据结构
        edge_src, edge_dst = edge_index[0], edge_index[1]
        
        # 2. 为每个原子构建邻居列表
        neighbor_lists = {}
        neighbor_offsets = {}
        
        for i in range(num_nodes):
            # 找到以原子i为起点的所有边
            mask = (edge_src == i)
            neighbors = edge_dst[mask]
            offsets = cell_offsets[mask]
            
            neighbor_lists[i] = neighbors.tolist()
            neighbor_offsets[i] = offsets.tolist()
        
        # 3. 适配PerCNet的get_new_3tuple逻辑
        # 为每个原子找到最近的两个邻居作为参考
        nearest_neighbors = {}
        for i in range(num_nodes):
            if len(neighbor_lists[i]) >= 2:
                neighbors = neighbor_lists[i]
                distances = []
                for j in neighbors:
                    # 计算距离（考虑周期性）
                    dist = torch.norm(pos[j] - pos[i]).item()
                    distances.append((dist, j))
                
                # 选择最近的两个邻居
                distances.sort()
                nearest_neighbors[i] = [distances[0][1], distances[1][1]]
            else:
                nearest_neighbors[i] = neighbor_lists[i] if neighbor_lists[i] else []
        
        # 4. 计算键角和二面角
        bond_angles_list = []
        dihedral_phi_list = []
        dihedral_tau_list = []
        angle_indices = []
        dihedral_indices = []
        
        for i in range(num_nodes):
            if len(neighbor_lists[i]) < 1:
                continue
                
            # 获取参考邻居
            ref_neighbors = nearest_neighbors[i]
            if len(ref_neighbors) < 2:
                continue
                
            n0_idx, n1_idx = ref_neighbors[0], ref_neighbors[1]
            
            # 计算参考向量
            pos_i = pos[i]
            pos_n0 = pos[n0_idx]
            pos_n1 = pos[n1_idx]
            
            vec_in0 = pos_n0 - pos_i
            vec_in1 = pos_n1 - pos_i
            
            for j in neighbor_lists[i]:
                pos_j = pos[j]
                vec_ij = pos_j - pos_i
                
                # 计算键角 theta (角度在vec_ij和vec_in0之间)
                dot_product = torch.dot(vec_ij, vec_in0)
                cross_product = torch.cross(vec_ij, vec_in0).norm()
                theta = torch.atan2(cross_product, dot_product)
                
                # 确保角度在[0, π]范围内
                if theta < 0:
                    theta = theta + math.pi
                if theta < 1e-5:
                    theta = torch.tensor(0.0)
                
                bond_angles_list.append(theta)
                angle_indices.append([i, j])
                
                # 计算二面角 phi (基于vec_in0和vec_in1的平面)
                dist_in0 = vec_in0.norm()
                if dist_in0 > 1e-8:  # 避免除零
                    plane1 = torch.cross(vec_ij, vec_in0)
                    plane2 = torch.cross(vec_in0, vec_in1)
                    
                    dot_planes = torch.dot(plane1, plane2)
                    cross_planes = torch.cross(plane1, plane2)
                    cross_dot_ref = torch.dot(cross_planes, vec_in0) / dist_in0
                    
                    phi = torch.atan2(cross_dot_ref, dot_planes)
                    if phi < 0:
                        phi = phi + math.pi
                    if phi < 1e-5:
                        phi = torch.tensor(0.0)
                else:
                    phi = torch.tensor(0.0)
                
                # 计算二面角 tau (更复杂的二面角)
                if j < len(neighbor_lists) and len(neighbor_lists[j]) > 0:
                    # 找到j的最近邻居作为参考
                    j_neighbors = neighbor_lists[j]
                    if j_neighbors:
                        k_j = j_neighbors[0]  # j的最近邻居
                        pos_kj = pos[k_j]
                        vec_jkj = pos_kj - pos_j
                        
                        dist_ji = vec_ij.norm()
                        if dist_ji > 1e-8:
                            plane1_tau = torch.cross(vec_ij, vec_in0)
                            plane2_tau = torch.cross(vec_ij, vec_jkj)
                            
                            dot_planes_tau = torch.dot(plane1_tau, plane2_tau)
                            cross_planes_tau = torch.cross(plane1_tau, plane2_tau)
                            cross_dot_ref_tau = torch.dot(cross_planes_tau, vec_ij) / dist_ji
                            
                            tau = torch.atan2(cross_dot_ref_tau, dot_planes_tau)
                            if tau < 0:
                                tau = tau + math.pi
                            if tau < 1e-5:
                                tau = torch.tensor(0.0)
                        else:
                            tau = torch.tensor(0.0)
                    else:
                        tau = torch.tensor(0.0)
                else:
                    tau = torch.tensor(0.0)
                
                dihedral_phi_list.append(phi)
                dihedral_tau_list.append(tau)
                dihedral_indices.append([i, j])
        
        # 5. 转换为张量
        if bond_angles_list:
            bond_angles = torch.stack(bond_angles_list)
            angle_index = torch.tensor(angle_indices).T.long()
        else:
            bond_angles = torch.tensor([])
            angle_index = torch.tensor([[], []]).long()
            
        if dihedral_phi_list:
            dihedral_phi = torch.stack(dihedral_phi_list)
            dihedral_tau = torch.stack(dihedral_tau_list)
            dihedral_index = torch.tensor(dihedral_indices).T.long()
        else:
            dihedral_phi = torch.tensor([])
            dihedral_tau = torch.tensor([])
            dihedral_index = torch.tensor([[], []]).long()
        
        return bond_angles, dihedral_phi, dihedral_tau, angle_index, dihedral_index
# ==================== 新增结束 ==================== #


class ToFloat(object):
    '''Converts all features in the crystall pattern graph to float'''
    def __call__(self, data):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.angle_attr = data.angle_attr.float()
        
        # ==================== 新增：处理二面角特征 ==================== #
        if hasattr(data, 'dihedral_phi') and data.dihedral_phi.numel() > 0:
            data.dihedral_phi = data.dihedral_phi.float()
        if hasattr(data, 'dihedral_tau') and data.dihedral_tau.numel() > 0:
            data.dihedral_tau = data.dihedral_tau.float()
        # ==================== 新增结束 ==================== #
        
        return data