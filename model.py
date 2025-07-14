#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   gcpnet.py
@Time    :   2023/04/03 09:10:32
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear,Sequential

from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool,DiffGroupNorm

import numpy as np
from typing import Literal, Optional
from baseModule import BaseModule, GCAO


class RBFExpansion(torch.nn.Module):
    r"""
    Expand interatomic distances with radial basis functions.
    Default: RBF Expansion on distances or angles to compute Gaussian distribution of embeddings.
    """

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)
        self.type = type

        if lengthscale is None:
            # SchNet-style: set lengthscales relative to granularity of RBF expansion
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance - self.centers.to(distance.device))
        switcher = {
            'gaussian': (-base ** 2).exp(),
            'quadratic': base ** 2,
            'linear': base,
            'inverse_quadratic': 1.0 / (1.0 + base ** 2),
            'multiquadric': (1.0 + base ** 2).sqrt(),
            'inverse_multiquadric': 1.0 / (1.0 + base ** 2).sqrt(),
            'spline': base ** 2 * (base + 1.0).log(),
            'poisson_one': (base - 1.0) * (-base).exp(),
            'poisson_two': (base - 2.0) / 2.0 * base * (-base).exp(),
            'matern32': (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp(),
            'matern52': (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp(),
        }
        result = switcher.get(self.type, None)
        if result.any():
            return result
        else:
            raise Exception("No Implemented Radial Basis Method")

class EmbeddingLayer(torch.nn.Module):
    r"""
    Embedding layer which performs nonlinear transform on atom, edge and triplet features
    """

    def __init__(self, input_features, output_features) -> None:
        super().__init__()

        self.mlp = Sequential(
            Linear(input_features, output_features),
            DiffGroupNorm(output_features, 6, track_running_stats=True),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GCPNetUpdate(torch.nn.Module):
    """
    Implementation of the GCGNNAT layer composed of EdgeGatedGraphConv steps
    """

    def __init__(self, hidden_features, dropout_rate) -> None:
        super().__init__()

        # Sequential GCPNet update layers
        # Overall mapping is input_features -> output_features
        self.bondAndAngleUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)
        self.bondAndAtomUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)

    def forward(
        self,
        g: Data,
        atom_feats: torch.Tensor,
        bond_attr: torch.Tensor,
        triplet_feats: torch.Tensor,
    ) -> torch.Tensor:
        # Perform sequential edge and node updates

        bond, triplet_feats = self.bondAndAngleUpdate(
            bond_attr,  g.angle_index, triplet_feats
        )

        atom_feats, bond_attr = self.bondAndAtomUpdate(atom_feats, g.edge_index, bond)

        # Return updated node, edge, and triplet embeddings
        return atom_feats, bond_attr, triplet_feats



class GCPNet(BaseModule):
    def __init__(self, 
                 data: Data, 
                 firstUpdateLayers: int=4,
                 secondUpdateLayers: int=4,
                 atom_input_features: int=106,
                 edge_input_features: int=50,
                 triplet_input_features: int=40,
                 # ==================== 新增参数 ==================== #
                 dihedral_input_features: int=40,  # 二面角特征维度
                 # ==================== 新增结束 ==================== #      
                 embedding_features: int=64,
                 hidden_features: int=256,
                 output_features: int=1,
                 min_edge_distance: float=0.0,
                 max_edge_distance: float=8.0,
                 min_angle: float=0.0,
                 max_angle: float=torch.acos(torch.zeros(1)).item() * 2,
                 link: Literal["identity", "log", "logit"] = "identity",
                 dropout_rate=0.0,
                ) -> None: 
        super().__init__()

        self.data = data  ##Todo: for future utilization

        self.atom_embedding = EmbeddingLayer(atom_input_features, hidden_features)

        self.edge_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=min_edge_distance, vmax=max_edge_distance, bins=edge_input_features
            ),
            EmbeddingLayer(edge_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        self.angle_embedding = torch.nn.Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=triplet_input_features),
            EmbeddingLayer(triplet_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        # ==================== 新增模块开始 ==================== #
        # 为新的3维无穷势能特征（库仑、伦敦、泡利）创建一个新的嵌入层
        self.inf_edge_embedding = torch.nn.Sequential(
            EmbeddingLayer(3, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )
        # 一个专用的GCAO层，用于处理无穷势能的全连接图
        self.infinite_update = GCAO(dim=hidden_features, dropout_rate=dropout_rate)
        
        # ===================== 新增模块结束 ===================== #

        # ==================== 新增：二面角嵌入层 ==================== #
        self.dihedral_embedding = torch.nn.Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=dihedral_input_features),
            EmbeddingLayer(dihedral_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )
        # ==================== 新增结束 ==================== #
        # layer to perform atom, bond and andle updates on the graph by 2N GCAO
        self.firstUpdate = torch.nn.ModuleList(
            [GCPNetUpdate(hidden_features,dropout_rate) for _ in range(firstUpdateLayers)]
        )

        # layer to perform atom and bond updates on the graph by N GCAO
        self.secondUpdate = torch.nn.ModuleList(
            [
                GCAO(dim=hidden_features,dropout_rate=dropout_rate)
                for _ in range(secondUpdateLayers)
            ]
        )

        # simple hign-level output layer
        self.fc = Linear(hidden_features, output_features)

        switcher = {
            "identity": self._identity,
            "log": torch.exp,
            "logit": torch.sigmoid
        }
        self.link = switcher.get(link, None)
        if link == "log":
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
 
    def _identity(self, x):
        return x
    
    @property
    def target_attr(self):
        """Specifies the target attribute property for writing output to file"""
        return "y"

    def forward(self, g: Data) -> torch.Tensor:
        # 解包局部图的数据 (已有代码)
        atom_feats = g.x
        bond_attr = g.edge_attr
        triplet_feats = g.angle_attr
                # ==================== 新增：处理二面角特征 ==================== #
        # 检查是否存在二面角特征
        has_dihedral = hasattr(g, 'dihedral_attr') and g.dihedral_attr.numel() > 0
        if has_dihedral:
            dihedral_feats = g.dihedral_attr
            dihedral_index = g.dihedral_index
        # ==================== 新增结束 ==================== #
        
        # ==================== 新增: 解包无穷图的数据 ==================== #
        inf_edge_index = g.inf_edge_index
        inf_edge_attr = g.inf_edge_attr
        # ================================================================= #

        # 进行初始嵌入 (已有代码)
        atom_feats = self.atom_embedding(atom_feats)
        bond_attr = self.edge_embedding(bond_attr)
        triplet_feats = self.angle_embedding(triplet_feats)
        # ==================== 新修正：二面角特征融合 ==================== #
        if has_dihedral:
            dihedral_feats = g.dihedral_attr
            dihedral_edge_indices = g.dihedral_index  # 现在这是边的索引
            
            # 嵌入二面角特征
            dihedral_embedded = self.dihedral_embedding(dihedral_feats)
            
            # 将二面角特征聚合到对应的边上
            from torch_scatter import scatter_add
            
            # 确保索引在有效范围内
            valid_mask = dihedral_edge_indices < bond_attr.size(0)
            if valid_mask.any():
                valid_dihedral_feats = dihedral_embedded[valid_mask]
                valid_edge_indices = dihedral_edge_indices[valid_mask]
                
                # 聚合到边特征上
                bond_enhancement = scatter_add(
                    valid_dihedral_feats, 
                    valid_edge_indices, 
                    dim=0, 
                    dim_size=bond_attr.size(0)
                )
                bond_attr = bond_attr + bond_enhancement
        # ================================================================= #

        # ==================== 新增: 嵌入无穷势能特征 ==================== #
        inf_edge_feats = self.inf_edge_embedding(inf_edge_attr)
        # ================================================================ #

        # 在 局部图 上进行GCPNet的序列更新 (已有代码)
        for update in self.firstUpdate:
            atom_feats, bond_attr, triplet_feats = update(
                g, atom_feats, bond_attr, triplet_feats
            )

        for update in self.secondUpdate:
            atom_feats, bond_attr = update(
                atom_feats, g.edge_index, bond_attr
            )
            
        # ==================== 新增: 势能信息融合 ==================== #
        
        # 在局部图更新后，将得到的原子特征作为无穷图更新的输入。
        # 这使得长程物理作用可以调整已经过局部几何信息优化的原子特征。
        infinite_atom_feats, _ = self.infinite_update(
            atom_feats, inf_edge_index, inf_edge_feats
        )
        
        # 融合来自局部图和无穷图的信息。简单的逐元素相加是一种有效的方式。
        final_atom_feats = atom_feats + infinite_atom_feats
        # ================================================================= #

        # 读出层 (修改为使用融合后的特征)
        out = global_mean_pool(final_atom_feats, g.batch)
        out = self.fc(out)


        ########################
        ###  interpretability
        # node_feats = self.fc(node_feats) 
        # out = global_mean_pool(node_feats, g.batch)
        # 
        # def atomic_number_to_symbol(atomic_number):
        #     from pymatgen.core.periodic_table import Element
        #     return Element.from_Z(atomic_number).symbol
        # 
        #
        # with open('node_contribs.csv','a') as f:
        #     # print('heat of formation [eV/atom], id, atomic symbol', file=f)
        #     for i in range(node_feats.shape[0]):
        #         print('{0},{1},{2}'.format(  torch.squeeze(node_feats, -1)[i].item(),
        #                                        g.AB[i] , 
        #                                        atomic_number_to_symbol(g.z.tolist()[i]))
        #                                        ,file=f)
        ##################################


        # apply link function
        if self.link:
            out = self.link(out)

        return torch.squeeze(out, -1)