"""
python v3.7.9
@Project: cation_new
@File   : csd_process.py
@Author : Zhiyuan Zhang
@Date   : 2022/5/11
@Time   : 15:46
"""
import os
import shutil
from itertools import product
from pathlib import Path
from typing import List, Union, Sequence, Callable
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from module.csd_miner import CSDMiner


class MetalLigandPairProcessor(object):
    """    This class is used to process crystal from CSD to Metal-Ligand Pairs    """
    def __init__(
            self,
            metals: List[str],
            dir_data: Path,
            path_identifier: Path,
            dir_anion_gas: Path,
            path_atomic_radii: Path,
            dataset_name: str = "dataset",
            dataset_type: type = None,
            dir_con_files: Path = None,
            coordinate_functional_groups: List[str] = None,
            sorted_coordinate_atoms: List[List[str]] = None,
            split_metal_ligand: bool = False,
            relative_bond_length: bool = True,
            to_filter: bool = True
    ):
        """
        Initialization.
        Args:
            metals: Scope of metals to Process
            dir_data: Primary folder directory
            path_identifier: Path of input identifier file
            dir_anion_gas: directory to save anion and gas molecules' files
            path_atomic_radii: Path of files to define the atomic radii
            dir_con_files: directory to save files.con that define the struct of functional groups
            coordinate_functional_groups: list of function group names that match with files.con's names,
                                          be used as coordinate functional groups when generate virtual pairs.
            sorted_coordinate_atoms: atomic symbol saving in inside list,
                                     whose order means their priority from highest to the lowest.
            split_metal_ligand: Whether to split Metal-Ligand pairs to metal and ligand
            to_filter: Whether to filter redundant Pairs
        """
        # Processed metals
        self.metals = metals
        self.coordinate_functional_groups = coordinate_functional_groups
        self.sorted_coordinate_atoms = sorted_coordinate_atoms
        self.split_metal_ligand = True if split_metal_ligand else False
        self.filter = to_filter
        self.dataset_type = dataset_type
        self.relative_bond_length = relative_bond_length

        # Input data path
        self.path_identifier = path_identifier
        self.dir_anion_gas = dir_anion_gas
        self.path_atomic_radii = path_atomic_radii
        self.dir_con_files = dir_con_files

        # Data save path
        self.dir_data = dir_data
        self.data_name = dir_data.stem
        self.dataset_name = dataset_name

    @property
    def dir_struct(self):
        return self.dir_data.joinpath("struct")

    @property
    def dir_pair(self):
        return self.dir_struct.joinpath("pair")

    @property
    def dir_bs(self):
        return self.dir_struct.joinpath("bs")

    @property
    def dir_ligand(self):
        return self.dir_struct.joinpath("ligand")

    @property
    def dir_virtual_pair(self):
        return self.dir_struct.joinpath("v_pair")

    @property
    def dir_dataset(self):
        return self.dir_data.joinpath(self.dataset_name)

    @property
    def dir_raw(self):
        return self.dir_dataset.joinpath("raw")

    @property
    def dir_mfv(self):
        return self.dir_raw.joinpath("mfv")

    @property
    def dir_ca(self):
        return self.dir_raw.joinpath("ca")

    def split_crystal_to_pairs(self, max_: int = None, **kwargs) -> None:
        """
        Split crystals in CSD to Metal-Ligand Pairs
        Args:
            max_: Debugging parameters use a small amount of data to test whether the whole program can run normally.
        """
        miner = CSDMiner()
        dict_id = miner.read_identifier_from_excel(self.path_identifier, *self.metals)
        for metal, list_id in dict_id.items():
            miner.get_entry(list_id[0: max_] if max_ else list_id)
            miner.remove_solvents_and_anion_from_entries(mol_file_dir=self.dir_anion_gas)
            miner.remove_isolated_atom_from_entry()
            miner.split_crystals_to_metal_ligands_pairs_(cm_symbol=metal)
            miner.write_bs_target(self.dir_pair, self.dir_bs)

    def _split_metal_ligand(self, csd_miner: CSDMiner):
        """
        Split pair to metal and ligand before the graph transform.
        The following operations are performed:
        1) Extract metal atomic feature vectors and then save to csv files
        2) Record and save coordinate atom's labels to csv files
        3) Remove metal from pairs saved in CSDMiner

        Initialize the class by give the saving files of metal atomic feature vectors and coordinate atom's labels.
        Then, Being Used as the input for func: graph_transform.
        """
        if not self.dir_ca.exists():
            self.dir_ca.mkdir(parents=True)
        if not self.dir_mfv.exists():
            self.dir_mfv.mkdir(parents=True)

        for pair in tqdm(csd_miner.list_target, "split metal ligand", colour="#00BBBB"):
            idt = pair.identifier

            # Extract metal atomic feature vector
            metal = [m for m in pair.atoms if m.is_metal][0]
            mfv = csd_miner.atomic_feature_vector(metal, True)  # metal feature vector
            ser_mfv = pd.Series(mfv, name=idt)
            ser_mfv.to_csv(self.dir_mfv.joinpath(f"{idt}.csv"))

            # Record coordinate atom's labels
            ca = [a.label for a in metal.neighbours]
            ser_ca = pd.Series(ca, name=idt, dtype=pd.Float64Dtype)
            ser_ca.to_csv(self.dir_ca.joinpath(f"{idt}.csv"))

            pair.remove_atom(metal)  # Remove metal atoms

    def remove_struct_data(self):
        """    Remove all structure files    """
        shutil.rmtree(self.dir_struct)

    def remove_raw_data(self):
        """    Remove all files and dirs in dataset raw directory    """
        shutil.rmtree(self.dir_raw)

    def remove_null_bs_ligand(self, nfs: List[str]):
        """    Try to remove bond info files and the Corresponding graph files when give bond info files' paths"""
        dirs = ["adjacency", "atomic_feature", "bond_feature", "ca", "mfv"]
        for d, nf in product(dirs, nfs):
            p = self.dir_raw.joinpath(f"{d}/{nf}")
            try:
                os.remove(p)
            except IOError:
                print(f"not found file: {p}")

    @property
    def raw_dirs(self) -> List[str]:
        """    The directories contained in dataset raw file    """
        return ["adjacency", "atomic_feature", "bond_feature", "target"]\
            + (["mfv", "ca"] if self.split_metal_ligand else [])

    def switch_dataset(self, dataset_name: str):
        """"""
        if dataset_name:
            if dataset_name not in ["dataset", "v_dataset", "rbl_dataset"]:
                raise ValueError(
                    "There only are there datasets named 'dataset', 'v_dataset' and 'rbl_dataset' can be choose,"
                    f"but got a dataset name: {dataset_name}"
                )
            self.dataset_name = dataset_name

    def graph_transform(
            self,
            ranges: Union[Sequence, range] = None,
            file_cdt: Callable = None,
            reader: str = "Molecule",
            file_type: str = "mol2",
            generator: bool = False,
            filter_out_inorganic=False,
            dataset_name: str = None,
            **kwargs
    ) -> List:
        """
        Performing graph transform
        :param file_cdt: the condition that a mol file will be selected to be transformed
        :param ranges: ranges of mols to be transformed
        :param file_type:
        :param reader: reader types, 'Molecule', 'Entry', or 'Crystal'
        :param filter_out_inorganic: whether to filter out inorganic molecules.
        :param generator:
        :param dataset_name: Change dataset name
        :return: CSDMiner class with read molecules information

        Args:
            dataset_name:
        """
        self.switch_dataset(dataset_name)
        if not self.dir_raw.exists():
            self.dir_raw.mkdir(parents=True)

        dir_pair = self.dir_pair if self.dataset_name == "dataset" else self.dir_virtual_pair
        dir_bs = self.dir_bs if self.dataset_name == "dataset" else self.dir_struct.joinpath("v_bs")
        csd_miner = CSDMiner()
        csd_miner.config_atom_radius_info(self.path_atomic_radii)
        csd_miner.list_target = csd_miner.file_reader(
            file_dir=dir_pair,
            file_type=file_type,
            generator=generator,
            reader=reader,
            ranges=ranges if ranges else None,
            condition=file_cdt if file_cdt else None
        )

        # target_transforms return
        if filter_out_inorganic:
            csd_miner.filter_organic()
        csd_miner.filter_mol_with_isolated_atom()

        if self.split_metal_ligand:
            self._split_metal_ligand(csd_miner)

        csd_miner.molecular_graph_transform(self.dir_raw)

        # Calculate median bond length
        null_files = csd_miner.get_bond_length_median(dir_bs, self.dir_raw.joinpath("target"), **kwargs)
        self.remove_null_bs_ligand(null_files)

        return null_files

    def get_pair_ligands(self):
        """"""
        miner = CSDMiner()
        miner.list_target = miner.file_reader(self.dir_pair, file_type="mol2")
        miner.get_ligands(n_bar=True)
        miner.put_mol_into_container()
        miner.list_target = miner.merge_same_pairs(miner.list_target)
        miner.get_mol_from_container()
        miner.file_writer(self.dir_ligand, file_type="mol2")

    def generate_virtual_pair(self):
        """"""
        miner = CSDMiner()
        miner.config_substructure_searchers(files_dir=self.dir_con_files)
        miner.list_target = miner.file_reader(self.dir_pair, file_type="mol2")
        miner.get_ligands()
        miner.put_mol_into_container()
        miner.list_target = miner.merge_same_pairs(miner.list_target)
        miner.get_mol_from_container()
        miner.file_writer(self.dir_ligand, file_type="mol2")

        miner.generate_virtual_pairs(
            self.metals,
            selected_fg=self.coordinate_functional_groups,
            sorted_ca=self.sorted_coordinate_atoms
        )
        miner.put_mol_into_container()
        miner.list_target = miner.merge_same_pairs(miner.list_target)
        miner.write_bs_target(self.dir_virtual_pair, self.dir_struct.joinpath("v_bs"))

    def match_graph_target(self) -> None:
        """    Remove any files whose name not exist in other files    """
        files = [
            set(os.listdir(self.dir_raw.joinpath(d))) for d in self.raw_dirs
        ]
        shared_files = set.intersection(*files)
        non_match_files = [f.difference(shared_files) for f in files]
        for i, d in enumerate(self.raw_dirs):
            path_dir = self.dir_raw.joinpath(d)
            for f in non_match_files[i]:
                pf = path_dir.joinpath(f)
                os.remove(pf)

    def cook_dataset(
            self,
            dataset: type,
            dataset_name: str = None,
            pre_filter: Callable = None,
            pre_transform: Callable = None,
            transform: Callable = None,
            remove_raw_data: bool = False,
            **kwargs
    ) -> Dataset:
        """"""
        self.switch_dataset(dataset_name)
        output = dataset(
            self.dir_dataset,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
            transform=transform,
            **kwargs
        )
        if self.dir_raw.exists() and remove_raw_data:
            self.remove_raw_data()

        return output

    def calculate_ideal_bond_length(self) -> None:
        """    Transform bond length info saved in raw dir to ideal bond length    """
        dir_target = self._copy_raw_data_to_rbl_raw_data()
        for path_csv in dir_target.glob("*.csv"):
            df = pd.read_csv(path_csv, index_col=0)
            idt = df.index
            col = df.columns
            data = df.values
            na = data[0, 0]
            for bond_idx in range(na):
                bond_lengths = data[:, 3 * bond_idx + 3]
                ideal_bond_lengths = data[:, 3 * bond_idx + 1]
                relative_bond_lengths = bond_lengths / ideal_bond_lengths
                data[:, 3 * bond_idx + 3] = relative_bond_lengths

            df = pd.DataFrame(data, index=idt, columns=col)
            df.to_csv(path_csv)

    def _copy_raw_data_to_rbl_raw_data(self):
        """    Copy bond length dataset raw data into relative bond length dataset raw    """
        # Remove directory tree if the destination exist
        dir_raw = self.dir_data.joinpath("dataset/raw")
        dir_ibl_raw = self.dir_data.joinpath("rbl_dataset/raw")
        if dir_ibl_raw.exists():
            shutil.rmtree(dir_ibl_raw)

        shutil.copytree(str(dir_raw), dir_ibl_raw)
        dir_target = dir_ibl_raw.joinpath("target")

        return dir_target

    def process(self, **kwargs):
        """"""
        self.split_crystal_to_pairs(**kwargs)
        self.graph_transform()
        self.match_graph_target()
        if not self.dataset_type:
            raise AttributeError("Attr: dataset type don't be set!")
        self.cook_dataset(self.dataset_type, "dataset", **kwargs)

        if self.relative_bond_length:
            self.calculate_ideal_bond_length()
            self.cook_dataset(self.dataset_type, "rbl_dataset", remove_raw_data=True, **kwargs)
            self.switch_dataset("dataset")
            self.remove_raw_data()

        if self.coordinate_functional_groups and self.sorted_coordinate_atoms and self.dir_con_files:
            self.generate_virtual_pair()
            self.graph_transform(dataset_name="v_dataset")
            self.match_graph_target()
            self.cook_dataset(self.dataset_type, remove_raw_data=True, **kwargs)

        self.remove_struct_data()


def graph_transform(
        dir_mol: Path, dir_ds_raw: Path, path_atomic_radii: Path,
        ranges: Union[Sequence, range] = None,
        file_cdt: Callable = None,
        reader: str = "Molecule",
        file_type: str = "mol2",
        generator: bool = False,
        filter_out_inorganic=False,
        target_transform: Callable = None
) -> (CSDMiner, object):
    """
    Performing graph transform
    :param dir_mol: directory of molecules come from
    :param dir_ds_raw: directory of dataset to save raw data
    :param file_cdt: the condition that a mol file will be selected to be transformed
    :param ranges: ranges of mols to be transformed
    :param file_type:
    :param reader: reader types, 'Molecule', 'Entry', or 'Crystal'
    :param target_transform: offer a method to transform data in CSDMiner.list_target
    :param filter_out_inorganic: whether to filter out inorganic molecules.
    :param generator:
    :param path_atomic_radii
    :return: CSDMiner class with read molecules information
    """
    if not dir_ds_raw.exists():
        dir_ds_raw.mkdir(parents=True)

    csd_miner = CSDMiner()
    csd_miner.config_atom_radius_info(path_atomic_radii)
    csd_miner.list_target = csd_miner.file_reader(
        dir_mol, file_type=file_type, generator=generator,
        reader=reader,
        ranges=ranges if ranges else None,
        condition=file_cdt if file_cdt else None
    )

    tt_return = None  # target_transforms return
    if filter_out_inorganic:
        csd_miner.filter_organic()
    csd_miner.filter_mol_with_isolated_atom()
    if target_transform:
        tt_return = target_transform(csd_miner)

    csd_miner.molecular_graph_transform(dir_ds_raw)

    return csd_miner, tt_return
