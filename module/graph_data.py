
"""
@Project: csd_miner.py
@File   : graph_data.py
@Author : Zhiyuan Zhang
@Date   : 2021/11/8 11:22
"""
import os
from abc import ABC
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Dataset, Data, InMemoryDataset


class BaseGraphDataset(Dataset, ABC):
    """    Dataset contain basic custom graph attributes    """

    @property
    def dir_target(self):
        return Path(self.raw_dir).joinpath("target")

    @property
    def dir_ad(self):
        return Path(self.raw_dir).joinpath("adjacency")

    @property
    def dir_af(self):
        return Path(self.raw_dir).joinpath("atomic_feature")

    @property
    def dir_bf(self):
        return Path(self.raw_dir).joinpath("bond_feature")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """"""
        return [str(n) for n in self.dir_target.glob("*.csv")]

    @property
    def raw_idt(self) -> Union[str, List[str], Tuple]:
        """"""
        return [p.stem for p in self.dir_target.glob("*.csv")]

    def process_base_graph(self, idt):
        """    Extracting from excel    """
        ad = pd.read_csv(self.dir_ad.joinpath(f"{idt}.csv"), index_col=0).values
        af = pd.read_csv(self.dir_af.joinpath(f"{idt}.csv"), index_col=0).values
        bf = pd.read_csv(self.dir_bf.joinpath(f"{idt}.csv"), index_col=0).values

        # Transfer tensors
        ad = torch.tensor(np.stack(np.nonzero(ad)), dtype=torch.float)
        af = torch.tensor(af.T, dtype=torch.float)
        bf = torch.tensor(bf, dtype=torch.float)

        return af, ad, bf


class BondLengthDataset(BaseGraphDataset, ABC):
    """"""
    def bond_target(self, idt):
        """    Extracting bond target    """
        target = pd.read_csv(self.dir_target.joinpath(f"{idt}.csv"), index_col=0).values.flatten()
        target = np.concatenate(
            [
                np.array([target[0]], dtype=float),
                np.array([target[[3 * i + 1, 3 * i + 3]] for i in range(target[0])], dtype=float).flatten()
            ]
        )
        target = torch.tensor(target, dtype=torch.float)

        return target


class Graph(InMemoryDataset, BondLengthDataset, ABC):
    """    Graph dataset with split Metal and Ligand    """
    def __init__(self, root: Union[Path, str], transform=None, pre_transform=None, pre_filter=None):
        """"""
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """"""
        return ["data.pt"]

    def process(self):
        """"""
        list_data = []
        # Transfer graph data in excel to standard PyG formal Data
        for idt in tqdm(self.raw_idt, colour="blue", leave=False):
            # Extracting from excel
            try:
                atom_feature, adjacency, bond_feature, mv, sorted_atoms = self.process_base_graph(idt)
            except:
                print(idt)
                continue
            target, ca = self.multi_bond_target(idt, sorted_atoms)

            if not target.shape[0]:
                continue

            # Generating graph formal data
            data = Data(atom_feature, adjacency, bond_feature, target, ca=ca, mv=mv, idt=idt)
            list_data.append(data)

        data, slices = self.collate(list_data)
        torch.save((data, slices), self.processed_paths[0])

    def process_base_graph(self, idt: str) -> (Tensor, Tensor, Tensor, Tensor, list):
        """    Extracting from excel    """
        ad = pd.read_csv(self.dir_ad.joinpath(f"{idt}.csv"), index_col=0)
        af = pd.read_csv(self.dir_af.joinpath(f"{idt}.csv"), index_col=0)
        bf = pd.read_csv(self.dir_bf.joinpath(f"{idt}.csv"), index_col=0)
        mv = pd.read_csv(self.dir_mv.joinpath(f"{idt}.csv"), index_col=0)

        sorted_atoms = list(af.columns)

        # Transfer tensors
        ad = torch.tensor(np.stack(np.nonzero(ad.values)), dtype=torch.float)
        af = torch.tensor(af.T.values, dtype=torch.float)
        bf = torch.tensor(bf.values, dtype=torch.float)
        mv = torch.tensor(mv.values.flatten(), dtype=torch.float).unsqueeze(0)  # metal feature vector

        return af, ad, bf, mv, sorted_atoms

    def multi_bond_target(self, idt: str, sorted_atoms: list):
        """"""
        target = pd.read_csv(self.dir_target.joinpath(f"{idt}.csv"), index_col=0).values.flatten()

        t_cal = target[2::3]  # coordinate atoms label
        t_ibl = target[1::3]  # ideal bond length
        t_bl = target[3::3]  # bond length

        cai = [sorted_atoms.index(lb) for lb in t_cal]
        array_cai = np.zeros(len(sorted_atoms), dtype=int)
        array_cai[cai] = 1

        array_cbl = np.zeros((len(t_cal), 1), dtype=float)
        for c, lbl in enumerate(t_cal):
            array_cbl[c][0] = t_bl[c]

        ca = torch.tensor(array_cai)
        target = torch.from_numpy(array_cbl)

        return target, ca

    @property
    def dir_ca(self):
        """"""
        return Path(self.raw_dir).joinpath("ca")

    @property
    def dir_mv(self):
        """"""
        return Path(self.raw_dir).joinpath("mfv")


class IMGraphDataset(InMemoryDataset, BondLengthDataset, ABC):
    """"""
    def __init__(self, root: Union[Path, str], transform=None, pre_transform=None, pre_filter=None, **kwargs):
        """"""
        super().__init__(root, transform, pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """"""
        return ["data.pt"]

    def process(self):
        """"""
        list_data = []
        # Transfer graph data in excel to standard PyG formal Data
        for idt in tqdm(self.raw_idt, colour="blue", leave=False):
            # Extracting from excel
            atom_feature, adjacency, bond_feature = self.process_base_graph(idt)
            target = self.bond_target(idt)
            # Generating graph formal data
            data = Data(atom_feature, adjacency, bond_feature, target, idt=idt)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            list_data.append(data)

        data, slices = self.collate(list_data)
        torch.save((data, slices), self.processed_paths[0])


class GraphDataset(BondLengthDataset, ABC):
    """"""
    def __init__(self, root: Union[Path, str], transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def __iter__(self):
        """"""

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """"""
        files = self.dir_target.glob("*.csv")
        return [f"{f.stem}.pt" for f in files]

    def process(self):
        """
        # Skipping process data when not raw data is not given
        if not self.path_graph_data:
            return
        :return:
        """

        # Transfer graph data in excel to standard PyG formal Data
        for idt in tqdm(self.raw_idt, colour="blue", leave=False):
            # Extracting from excel
            atom_feature, adjacency, bond_feature = self.process_base_graph(idt)
            target = self.bond_target(idt)
            # Generating graph formal data
            data = Data(atom_feature, adjacency, bond_feature, target, idt=idt)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir).joinpath(f"{idt}.pt"))

    def get(self, idx: int) -> Data:
        """"""
        return torch.load(self.processed_paths[idx])

    def len(self) -> int:
        """"""
        return len(self.processed_file_names)


class CookedDataset(BaseGraphDataset, ABC):
    """    a Dataset is not need to precess    """
    def __init__(self, root: Union[Path, str], transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """"""
        files = Path(self.processed_dir).glob("*.pt")
        return [f"{f.stem}.pt" for f in files if f.stem not in ["pre_transform", "pre_filter"]]

    def get(self, idx: int) -> Data:
        """"""
        return torch.load(self.processed_paths[idx])

    def len(self) -> int:
        """"""
        return len(self.processed_file_names)


class SelectYDescriptor(object):

    def __init__(self, *extract_idx):
        """"""
        self.extract_idx = extract_idx

    def __call__(self, data: Data):
        """
        Select descriptors of y in Data
        :param data: input graph data
        :param extract_idx: selected index of descriptor
        :return: graph data after the transferring
        """
        # Parameter check
        if not self.extract_idx:
            raise ValueError("extract_idx is not assign")

        y = data.y.unsqueeze(0)
        y = y[:, self.extract_idx]
        y = y.squeeze()
        data.y = y.unsqueeze(0)

        return data


class BondNumberFilters(object):

    def __init__(self, min_value=None, max_value=None):
        """"""
        self.min = min_value
        self.max = max_value

    def __call__(self, data: Data, *args, **kwargs):
        """"""
        y = data.y
        if self.min and y[0] < self.min:
            return False
        if self.max and y[0] > self.max:
            return False
        return True
