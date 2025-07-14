#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import numpy as np
import math
from torch_sparse import coalesce
from utils.helpers import compute_bond_angles, triplets

class GetY(object):
    '''Specify target for prediction'''
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data

class GetPeriodicGeometry(object):
    '''Computes bond angles and dihedral angles in the crystal using PerCNet method'''
    
    def __call__(self, data):
        # 计算键角和二面角
        angles, angle_index, dihedral_attrs, dihedral_edge_indices = self.compute_periodic_geometry(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        
        # 存储结果
        data.angle_attr = angles.reshape(-1, 1)
        data.angle_index = angle_index
        data.dihedral_attr = dihedral_attrs.reshape(-1, 1)
        data.dihedral_index = dihedral_edge_indices  # 现在这是边的索引，不是原子对
        
        return data
    
    def compute_periodic_geometry(self, pos, cell_offsets, edge_index, num_nodes):
        """
        计算键角和二面角，确保索引映射正确
        """
        # 1. 使用GCPNet现有的triplets函数获取三元组
        idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, cell_offsets, num_nodes)
        
        # 2. 计算键角（保持与原GCPNet兼容）
        pos_i = pos[idx_i]
        pos_j = pos[idx_j] 
        pos_k = pos[idx_k]
        
        # 考虑周期性边界条件
        pos_ji = pos_i - pos_j + cell_offsets[idx_ji]
        pos_jk = pos_k - pos_j + cell_offsets[idx_kj]
        
        # 计算夹角
        a = (pos_ji * pos_jk).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_jk).norm(dim=-1)
        angles = torch.atan2(b, a)
        
        # 角度索引：使用ji和kj边的索引
        angle_index = torch.stack([idx_kj, idx_ji], dim=0)
        
        # 3. 计算二面角
        dihedral_list = []
        dihedral_edge_idx_list = []
        
        # 构建边索引到边ID的映射
        edge_to_id = {}
        for edge_id, (src, dst) in enumerate(edge_index.T):
            edge_to_id[(src.item(), dst.item())] = edge_id
        
        # 构建邻接列表
        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
        for src, dst in edge_index.T:
            adj_list[src.item()].append(dst.item())
        
        # 对每个三元组(i,j,k)，寻找第四个原子l构成四元组
        for triplet_idx in range(len(idx_i)):
            i, j, k = idx_i[triplet_idx].item(), idx_j[triplet_idx].item(), idx_k[triplet_idx].item()
            
            # 寻找与j相连且不是i或k的原子l
            for l in adj_list[j]:
                if l != i and l != k:
                    # 计算二面角
                    dihedral = self._calculate_dihedral_angle(pos, i, j, k, l, cell_offsets, idx_ji[triplet_idx], idx_kj[triplet_idx])
                    
                    if dihedral is not None:
                        dihedral_list.append(dihedral)
                        
                        # 关键修改：找到中心边j->k在edge_index中的位置
                        if (j, k) in edge_to_id:
                            center_edge_id = edge_to_id[(j, k)]
                        elif (k, j) in edge_to_id:
                            center_edge_id = edge_to_id[(k, j)]
                        else:
                            continue  # 如果找不到对应的边，跳过
                        
                        dihedral_edge_idx_list.append(center_edge_id)
                    break  # 只取第一个找到的l
        
        # 转换为张量
        if dihedral_list:
            dihedral_attrs = torch.stack(dihedral_list)
            dihedral_edge_indices = torch.tensor(dihedral_edge_idx_list, dtype=torch.long)
        else:
            dihedral_attrs = torch.zeros(0)
            dihedral_edge_indices = torch.zeros(0, dtype=torch.long)
        
        return angles, angle_index, dihedral_attrs, dihedral_edge_indices
    
    def _calculate_dihedral_angle(self, pos, i, j, k, l, cell_offsets, ji_offset_idx, kj_offset_idx):
        """
        计算四原子二面角：沿着j-k键的扭转角
        """
        try:
            # 获取原子位置（考虑周期性）
            pos_i = pos[i] + cell_offsets[ji_offset_idx]
            pos_j = pos[j]
            pos_k = pos[k] + cell_offsets[kj_offset_idx]
            pos_l = pos[l]  # 假设l在同一晶胞
            
            # 构建向量
            v1 = pos_i - pos_j  # j->i
            v2 = pos_k - pos_j  # j->k (中心键)
            v3 = pos_l - pos_k  # k->l
            
            # 计算两个平面的法向量
            n1 = torch.cross(v1, v2)  # 平面(i,j,k)的法向量
            n2 = torch.cross(v2, v3)  # 平面(j,k,l)的法向量
            
            # 计算二面角
            cos_dihedral = torch.dot(n1, n2) / (torch.norm(n1) * torch.norm(n2) + 1e-8)
            cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)
            dihedral = torch.acos(cos_dihedral)
            
            return dihedral
        except:
            return None

class ToFloat(object):
    '''Converts all features to float'''
    def __call__(self, data):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.angle_attr = data.angle_attr.float()
        
        if hasattr(data, 'dihedral_attr') and data.dihedral_attr.numel() > 0:
            data.dihedral_attr = data.dihedral_attr.float()
        
        return data