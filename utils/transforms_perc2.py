#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   transforms.py
@Time    :   2023/05/05 12:41:00
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.eddu.com
'''
import torch
import math
import numpy as np
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

# ==================== 修改：用新类替代旧的GetAngle ==================== #
class GetPeriodicGeometry(object):
    '''Computes bond angles and dihedral angles in the crystal'''
    
    def __call__(self, data):
        # 1. 计算传统的键角（保持兼容性）
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        data.angle_index = torch.stack([idx_kj, idx_ji], dim=0)
        data.angle_attr = angles.reshape(-1, 1)
        
        # 2. 计算二面角（新增功能）
        dihedral_attrs, dihedral_indices = self._compute_dihedral_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        data.dihedral_attr = dihedral_attrs
        data.dihedral_index = dihedral_indices
        
        return data
    
    def _compute_dihedral_angles(self, pos, cell_offsets, edge_index, num_nodes):
        """
        基于PerCNet思想计算二面角，但适配GCPNet的数据结构
        """
        from utils.helpers import triplets
        
        # 获取三元组信息（借用GCPNet现有函数）
        idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, cell_offsets, num_nodes)
        
        # 为每个三元组(i,j,k)寻找第四个原子l，构成四元组(i,j,k,l)
        # 这里使用一个简化的策略：对每个j，找其邻居中除了i,k之外的原子
        edge_src, edge_dst = edge_index[0], edge_index[1]
        
        dihedral_list = []
        dihedral_idx_list = []
        
        # 构建邻接字典以快速查找邻居
        adj_dict = {}
        for src, dst in zip(edge_src.tolist(), edge_dst.tolist()):
            if src not in adj_dict:
                adj_dict[src] = []
            adj_dict[src].append(dst)
        
        for triplet_idx in range(len(idx_i)):
            i, j, k = idx_i[triplet_idx].item(), idx_j[triplet_idx].item(), idx_k[triplet_idx].item()
            
            # 寻找与j相连但不是i或k的原子l
            if j in adj_dict:
                for l in adj_dict[j]:
                    if l != i and l != k:
                        # 计算四元组(i,j,k,l)的二面角
                        dihedral = self._calculate_dihedral(pos, i, j, k, l, cell_offsets, idx_kj[triplet_idx], idx_ji[triplet_idx])
                        if dihedral is not None:
                            dihedral_list.append(dihedral)
                            # 使用中心键(i,j)作为二面角的索引参考
                            edge_idx = self._find_edge_index(edge_index, i, j)
                            if edge_idx is not None:
                                dihedral_idx_list.append(edge_idx)
                        break  # 只取第一个找到的l原子
        
        if dihedral_list:
            dihedral_attrs = torch.stack(dihedral_list).reshape(-1, 1)
            dihedral_indices = torch.tensor(dihedral_idx_list, dtype=torch.long)
        else:
            # 如果没有找到二面角，创建空张量
            dihedral_attrs = torch.zeros((0, 1), dtype=torch.float)
            dihedral_indices = torch.zeros((0,), dtype=torch.long)
            
        return dihedral_attrs, dihedral_indices
    
    def _calculate_dihedral(self, pos, i, j, k, l, cell_offsets, kj_offset_idx, ji_offset_idx):
        """
        计算四个原子i,j,k,l构成的二面角
        """
        try:
            # 获取原子位置
            pos_i = pos[i]
            pos_j = pos[j] + cell_offsets[ji_offset_idx]  # 考虑周期性
            pos_k = pos[k] + cell_offsets[kj_offset_idx]  # 考虑周期性
            pos_l = pos[l]  # 假设l在同一个晶胞内
            
            # 计算向量
            v1 = pos_i - pos_j
            v2 = pos_k - pos_j  
            v3 = pos_l - pos_j
            
            # 计算法向量
            n1 = torch.cross(v1, v2)
            n2 = torch.cross(v2, v3)
            
            # 计算二面角
            cos_angle = torch.dot(n1, n2) / (torch.norm(n1) * torch.norm(n2) + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            dihedral = torch.acos(cos_angle)
            
            return dihedral
        except:
            return None
    
    def _find_edge_index(self, edge_index, src, dst):
        """
        找到边(src,dst)在edge_index中的位置
        """
        edge_src, edge_dst = edge_index[0], edge_index[1]
        mask = (edge_src == src) & (edge_dst == dst)
        indices = torch.where(mask)[0]
        return indices[0].item() if len(indices) > 0 else None

# ==================== 修改结束 ==================== #

class ToFloat(object):
    '''Converts all features in the crystall pattern graph to float'''
    def __call__(self, data):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.angle_attr = data.angle_attr.float()
        
        # ==================== 新增：处理二面角特征 ==================== #
        if hasattr(data, 'dihedral_attr'):
            data.dihedral_attr = data.dihedral_attr.float()
        # ==================== 新增结束 ==================== #
        
        return data