"""
python v3.7.9
@Project: code
@File   : models.py
@Author : Zhiyuan Zhang
@Date   : 2021/11/16
@Time   : 15:32
"""
from typing import Dict, Sequence
import torch
from torch.nn import functional as fn
from torch.nn import Linear, Module, ModuleList, MultiheadAttention, Sigmoid, Parameter
from torch_geometric.nn import GraphConv, ASAPooling, JumpingKnowledge, global_mean_pool
from torch_geometric.data import Dataset


class GraphFeatureExtract(Module):
    """"""
    def __init__(
            self,
            gcn_layers: int,
            dataset: Dataset,
            hidden_size: int = 22,
            gcn: type = GraphConv,
            gpl: type = ASAPooling,
            pools: Dict = None,
            aggr: str = "mean"
    ) -> None:
        """"""
        super(GraphFeatureExtract, self).__init__()
        neurons = [dataset.num_features] + [hidden_size for _ in range(0, gcn_layers)]
        input_neuron = neurons[0]
        self.pool_loc = list(pools.keys()) if pools else []
        self.cvs = torch.nn.ModuleList()
        self.pls = torch.nn.ModuleList()
        self.out_channel = 0  # The total number of dimension for extracted feature vector
        for idx, output_neuron in enumerate(neurons[1:]):
            self.cvs.append(gcn(input_neuron, output_neuron, aggr=aggr))
            if idx in self.pool_loc:
                self.pls.append(gpl(output_neuron, **pools[idx]))
                self.out_channel += output_neuron
            input_neuron = output_neuron  # Update the input_neuron

        self.jump = JumpingKnowledge(mode="cat")
        self.out_channel += input_neuron

    def forward(self, batch):
        """"""
        x, edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
        edge_idx = edge_idx.to(torch.long)
        x = x.to(torch.float)
        edge_weight = None
        pl_idx = 0  # pooling layer indexes
        readout_features = []  # To record readout features
        for idx, cv in enumerate(self.cvs, 1):
            x = fn.relu(cv(x, edge_idx, edge_weight=edge_weight))  # Message passage
            if idx in self.pool_loc:
                readout_features.append(global_mean_pool(x, batch_idx))  # Readout
                # Pooling
                pl = self.pls[pl_idx]
                x, edge_idx, edge_weight, batch_idx, _ = pl(
                    x=x, edge_index=edge_idx, edge_weight=edge_weight,
                    batch=batch_idx)
                pl_idx += 1

        readout_features.append(global_mean_pool(x, batch_idx))  # Final readout
        # Concatenate
        feature = self.jump(readout_features)

        return feature

    @property
    def name(self):
        return self.__class__.__name__


class FullConnectNN(Module):
    """"""
    def __init__(self, neurons: Sequence[int], loc_dropout: Dict = None,
                 lin_fn=Linear, drop_fn=fn.dropout, act_fn=fn.relu):

        # Parameter check
        if not neurons:
            raise ValueError("the Graph Convolution Layers are not assigned!")
        if len(neurons) < 2:
            raise ValueError("Two value of num_neurons need to be given at least!")
        if not drop_fn and loc_dropout:
            raise ValueError("the value of arg drop_fn is not given!")

        if loc_dropout and isinstance(loc_dropout, Dict):
            if any(not isinstance(k, int) for k in loc_dropout.keys()):
                raise TypeError("all of keys in loc_dropout must be integer!")
            if max(loc_dropout.keys()) > len(neurons) - 2:
                raise ValueError("the maximum of loc_dropout must list than value of length of num_neurons minus 2")

            if all(isinstance(v, Dict) for v in loc_dropout.values()):
                if any(not isinstance(lk, str) for v in loc_dropout.values() for lk in v.keys()):
                    raise TypeError("the args of pools layers are assigned by Dicts, "
                                    "with str keys and int or float values")
            else:
                raise TypeError("all of values in loc_dropout are expected to be dict!")

        elif loc_dropout:
            raise TypeError(f"arg loc_dropout only allow Sequence type, but received a {type(loc_dropout)} instead!")

        super(FullConnectNN, self).__init__()
        input_neuron = neurons[0]
        self.lns = torch.nn.ModuleList()
        self.dpt = drop_fn
        self.act = act_fn
        self.loc_dropout = loc_dropout
        for idx, output_neuron in enumerate(neurons):
            self.lns.append(lin_fn(input_neuron, output_neuron))
            input_neuron = output_neuron

        self.out_channel = neurons[-1]

    def forward(self, feature_vector: torch.Tensor):
        """"""
        x = feature_vector
        for idx, ln in enumerate(self.lns):
            x = self.act(ln(x))
            if idx in self.loc_dropout.keys():
                x = self.dpt(x, training=self.training, **self.loc_dropout[idx])

        return x

    @property
    def name(self):
        """"""
        return self.__class__.__name__


class FCNCell(Module):
    """"""
    def __init__(
            self, in_feature: int, out_feature: int,
            activate_fn: type = torch.nn.ReLU, dropout_p=0.5, dropout_training=True
    ):
        super(FCNCell, self).__init__()
        self.lin = Linear(in_feature, out_feature)
        self.sigma = activate_fn()
        self.p = dropout_p
        self.training = dropout_training

    def forward(self, x):
        """"""
        x = fn.dropout(x, self.p, self.training)
        x = self.lin(x)
        x = self.sigma(x)

        return x


class Model(Module):
    """    assembling completely to training    """
    def __init__(
            self,
            gcn_layers: int,
            fcn_layers: int,
            dataset: Dataset,
            pool_local: Sequence[int],
            pool_ratio1: float = 0.65,
            pool_ratio2: float = 0.5,
            hidden_size: int = 22,
            gcn: type = GraphConv,
            gpl: type = ASAPooling,
            aggr: str = "mean"
    ):
        super(Model, self).__init__()
        dict_fcn_neurons = {
            2: [66, 1],
            3: [66, 30, 1],
            4: [66, 30, 12, 1],
            5: [66, 40, 20, 10, 1]
        }
        self.fe = GraphFeatureExtract(
            gcn_layers=gcn_layers,
            dataset=dataset,
            hidden_size=hidden_size,
            gcn=gcn,
            gpl=gpl,
            aggr=aggr,
            pools={
                pool_local[0]: {"ratio": pool_ratio1, "dropout": 0.5},
                pool_local[1]: {"ratio": pool_ratio2, "dropout": 0.5},
            }
        )
        fcn_neurons = dict_fcn_neurons[fcn_layers]
        self.fcn = ModuleList(
            FCNCell(i, o, activate_fn=torch.nn.LeakyReLU) for i, o in zip(fcn_neurons[:-1], fcn_neurons[1:])
        )

    def forward(self, batch):
        f = self.fe(batch)  # latent space
        for fcn in self.fcn:
            f = fcn(f)

        return f
