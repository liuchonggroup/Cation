"""
@Python : V 3.7.6
@Project: code
@File   : csd_miner.py
@Author : Zhiyuan Zhang
@Date   : 2021/10/25 13:57

Based on CCDC Python API v2021.02
"""
import os
import sys
from pathlib import Path
from typing import List, Union, Dict, Set, Sequence, Callable, Container, Generator
from functools import cmp_to_key
import numpy as np
import pandas as pd
from tqdm import tqdm
from ccdc import search, io
from ccdc.crystal import Crystal
from ccdc.diagram import DiagramGenerator
from ccdc.entry import Entry
from ccdc.molecule import Molecule, Atom
from ccdc.conformer import MoleculeMinimiser, ConformerGenerator

from module.zzytool import p_bar


def nothing_bar(data, *args, **kwargs):
    """    Do Nothing    """
    return data


class CSDMiner(object):
    """    The comprehensive Class to operate CSD, that is a Miner    """

    def __init__(self):
        """    Initialization    """
        self.list_target = []  # To store CSD instance, Entry, Crystal, Molecule
        self.searchers = {}  # To store searchers
        self.dict_bs = {}  # To store bonds' structure for pairs with a specific graph information, 2D list
        self.atomic_radius = None  # Covalent radius

    def atomic_feature_vector(self, atom: Atom, outermost_layer: bool) -> Dict:
        """
        Calculating atomic features
        :param atom: an Atom class
        :param outermost_layer: Whether get outermost open shell electron's structure, or instead of
                                getting all of electron's structure, as a atom's orbital feature.
        :return: The feature vector of the input atom
        """
        atom_feature = {}
        # Calculating orbital feature
        orbital_feature = self._atomic_orbital_feature(atom, outermost_layer)
        atom_feature.update(orbital_feature)
        # Get atomic covalent radius
        if self.atomic_radius:
            try:
                atom_feature["atomic_radius"] = self.atomic_radius[atom.atomic_symbol]
            except:
                atom_feature["atomic_radius"] = 0
        else:
            raise AttributeError("atomic radius was not configured!!")
        # Whether the atom is a metal
        atom_feature["is_metal"] = int(atom.is_metal)
        # Calculating feature about the rings
        atom_rings_feature = {
            "is_aromatic": 0,  # Whether the atoms on aromatic rings
            3: 0,  # How many 3-members rings is the atom on
            4: 0,  # How many 4-members rings is the atom on
            5: 0,  # How many 5-members rings is the atom on
            6: 0,  # How many 6-members rings is the atom on
            7: 0,  # How many 7-members rings is the atom on
            8: 0  # How many 8-members rings is the atom on
        }
        if atom.is_cyclic:
            rings = atom.rings
            if any(ring.is_aromatic for ring in rings):
                atom_rings_feature["is_aromatic"] = 1
            list_ring_member = [len(ring.bonds) for ring in rings]
            for rmb in list_ring_member:  # for each ring members
                if rmb in [3, 4, 5, 6, 7, 8]:
                    atom_rings_feature[rmb] += 1
        # Update atom feature
        atom_feature.update(atom_rings_feature)

        return atom_feature

    @staticmethod
    def _atomic_orbital_feature(atom, outermost_layer=True):
        """    Calculating the feature about atomic orbital structures    """
        _atomic_orbital_structure_max = {"1s": 2,
                                         "2s": 2, "2p": 6,
                                         "3s": 2, "3p": 6,
                                         "4s": 2, "3d": 10, "4p": 6,
                                         "5s": 2, "4d": 10, "5p": 6,
                                         "6s": 2, "4f": 14, "5d": 10, "6p": 6,
                                         "7s": 2, "5f": 14, "6d": 10, "7p": 6
                                         }
        atomic_orbital_structure = {"1s": 0,
                                    "2s": 0, "2p": 0,
                                    "3s": 0, "3p": 0,
                                    "4s": 0, "3d": 0, "4p": 0,
                                    "5s": 0, "4d": 0, "5p": 0,
                                    "6s": 0, "4f": 0, "5d": 0, "6p": 0,
                                    "7s": 0, "5f": 0, "6d": 0, "7p": 0}

        # Calculating atomic orbital structure
        residual_electron = atom.atomic_number
        n_osl = 0  # Principal quantum number (n) of open shell layers (osl)
        for orbital_name, men in _atomic_orbital_structure_max.items():  # max electron number (men)
            if residual_electron - men >= 0:
                residual_electron = residual_electron - men
                atomic_orbital_structure[orbital_name] = men
                if orbital_name[1] == "s":
                    n_osl = int(orbital_name[0])
            else:
                atomic_orbital_structure[orbital_name] = residual_electron
                break

        # Readout and return outermost electron structure
        atom_orbital_feature = {"atomic_number": atom.atomic_number, "n_osl": n_osl}
        if outermost_layer:
            diff_max_n = {"s": 0, "p": 0, "d": -1, "f": -2}
            for layer, diff in diff_max_n.items():  # Angular momentum quantum number (l)
                electron_number = atomic_orbital_structure.get(f"{n_osl + diff}{layer}", 0)
                atom_orbital_feature[layer] = electron_number
        else:
            atom_orbital_feature.update(atomic_orbital_structure)

        # return whole electron structure directly
        return atom_orbital_feature

    @staticmethod
    def _bond_feature_matrix(mol, mol_atoms_lbl, adjacency, bond_feature_name, int_bt=True):
        """
        Calculating bond features
        :param mol: Input molecule to calculate bond feature matrix
        :param mol_atoms_lbl: list of atom's labels for the mol
        :param adjacency: the graph adjacency of mol
        :param bond_feature_name: list of attribute's names to get bond's descriptor by calling the name directly
        :return: the bonds matrix, the bonds' names
        """

        # For each bond get the indexes of first atom in the start end and second atom in the final end
        def _bond_type_transform(bt: Union[str, int]):
            """"""
            if int_bt:
                bt = str(bt)
            else:
                bt = int(bt)

            if isinstance(bt, (str, int)):
                if int_bt and isinstance(bt, str):
                    return bt_format[bt]
                elif isinstance(bt, int):
                    return bt_format[bt]
                return bt
            else:
                raise ValueError(f"{type(bt)} are not allowed for bond type!")

        if int_bt:
            bt_format = {'Unknown': 0, 'Single': 1, 'Double': 2, 'Triple': 3, 'Quadruple': 4,
                         'Aromatic': 5, 'Delocalised': 7, 'Pi': 9}
        else:
            bt_format = {0: "Unknown", 1: "Single", 2: "Double", 3: "Triple", 4: "Quadruple",
                         5: "Aromatic", 7: "Delocalised", 9: "Pi"}

        bond_feature_matrix = []
        bond_name = []
        for b_row, b_col in zip(*np.nonzero(adjacency)):
            atom1_lbl, atom2_lbl = mol_atoms_lbl[b_row], mol_atoms_lbl[b_col]
            bond = mol.bond(atom1_lbl, atom2_lbl)
            bond_name.append(f"{atom1_lbl}-{atom2_lbl}")
            bond_feature = [int(eval(f"bond.{fn}", {"bond": bond}))
                            if isinstance(eval(f"bond.{fn}", {"bond": bond}), bool)
                            else eval(f"bond.{fn}", {"bond": bond})
                            for fn in bond_feature_name
                            ]

            # Transfer bond type symbol if it is in the bond feature
            if "bond_type" in bond_feature_name:
                idx = bond_feature_name.index("bond_type")
                bond_feature[idx] = _bond_type_transform(bond_feature[idx])

            bond_feature_matrix.append(bond_feature)

        return bond_feature_matrix, bond_name

    @staticmethod
    def _cm_symbol(cm_symbol):
        """    a transform for cm_symbol    """
        cm_symbol = cm_symbol if isinstance(cm_symbol, list) else [cm_symbol]
        is_all_metals = True if any(cms in ["any", "metal", None] for cms in cm_symbol) else False
        return cm_symbol, is_all_metals

    @staticmethod
    def _coordination_metal_with_their_coordination_atoms(mol, cm_symbol, is_all_metals):
        """    Get labels of coordination metals (cm) and the set of their coordination atoms    """
        return {m.label: {a.label for a in m.neighbours}
                for m in mol.atoms
                if m.is_metal and (is_all_metals or m.atomic_symbol in cm_symbol)
                }

    def _molecular_graph_adjacency_and_feature_matrix(self, mol, outermost_layer):
        """
        Calculating atoms' labels in the molecule.
        Calculating molecular graph adjacency matrix.
        Calculating molecular feature matrix.
        :param mol: a Molecule class
        :param outermost_layer: Whether get outermost open shell electron's structure, or instead of
                                getting all of electron's structure, as a atom's orbital feature.
        :return: atoms' labels, adjacency matrix, feature matrix
        """
        atom_feature_matrix = {}
        mol_atoms_lbl = [a.label for a in mol.atoms]
        num_atoms = len(mol_atoms_lbl)
        adjacency = np.zeros((num_atoms, num_atoms))
        for row, atom_lbl in enumerate(mol_atoms_lbl):
            atom = mol.atom(atom_lbl)

            # Calculating the adjacency of an atom
            for nbr_atom in atom.neighbours:
                column = mol_atoms_lbl.index(nbr_atom.label)
                adjacency[row][column] = 1

            # Calculating atomic features
            atomic_feature = self.atomic_feature_vector(atom, outermost_layer)
            # Assemble as atom feature matrix
            atom_feature_matrix[atom_lbl] = atomic_feature

        return mol_atoms_lbl, adjacency, atom_feature_matrix

    def add_attributes_into_entry(self, dict_attrs: Dict, report_skip=True):
        """
        Add attributes into entries by giving a dict of attributes.
        :param report_skip: if True, print identifiers of entries skipped to add attributes
        :param dict_attrs: the keys are entries identifier,
                           the values are a dict
                           with the keys of attributes' names and corresponding values
        :return:
        """
        for entry in self.list_target:
            dict_attr = dict_attrs.get(entry.identifier, None)

            # Skip if not attributes are here
            if not dict_attr:
                if report_skip:
                    print(entry.identifer)
                continue

            for attr_name, value in dict_attr.items():
                entry.attributes[attr_name] = value

    def bond_structure_writer(self, dir_bs_save: Path, dict_bs: Dict = None):
        """
        Sort out the order of bond structure information and save them:
        1) the problematic bond's information is discarded;
        2) equivalent coordination bond on the diagram information are put in the same column,
        3) save the sorted information to csv files
        :param dir_bs_save: dir to save bond structure information
        :param dict_bs: input bond structure information
        :return:
        """

        # Check if the save dir is exist
        if not dir_bs_save.exists():
            dir_bs_save.mkdir(parents=True)

        # Determining data source
        dict_bs = dict_bs if dict_bs else self.dict_bs

        # Sort out the order of bond structure information
        # TODO
        dict_error = {}  # recording error bonds to report
        for idt, (list_pop_idt, list_bs) in tqdm(
                dict_bs.items(), "Writing bond structures", colour="blue", leave=False
        ):
            # Get number of bonds (nb)
            nb = list_bs[0][0]

            # Filter out problematic bond's information
            list_temp = []
            list_temp_idt = []
            for idx, bs in enumerate(list_bs):
                if bs[0] == nb:
                    list_temp.append(bs)
                    list_temp_idt.append(list_pop_idt[idx])
            list_bs = list_temp
            list_pop_idt = list_temp_idt

            # Order bonds' info as the order in the first pair
            # Extract the bond order first in first pair
            first_pair_ca = [list_bs[0][2 + 3 * i] for i in range(nb)]
            # Split each bond info into the format that each a single bond in a unique list.
            other_pair_bonds = [[list_pair_bs[1 + 3 * j:4 + 3 * j] for j in range(nb)] for list_pair_bs in list_bs[1:]]
            # marching bond info (pbi) for each pairs except the first
            list_discard = []
            for idx, pbi in enumerate(list_bs[1:], start=0):
                pbi = [nb]  # Reinitialize of bond's info
                discard = False

                # Extract ca symbol as the order in first pair
                for ca_sbl in first_pair_ca:
                    other_pair_bond = other_pair_bonds[idx]

                    for c, a_bond in enumerate(other_pair_bond):
                        if a_bond[1] == ca_sbl:
                            pbi.extend(other_pair_bond.pop(c))
                            break

                        # if do not find any ca match with ca_sbl in other bonds, raise error
                        if c == len(other_pair_bond) - 1:
                            list_discard.append(idx + 1)
                            discard = True
                            print(f"\rbond info error: {idt}")

                    if discard:
                        break

            # Discard error data
            if list_discard:
                for idx in sorted(list_discard, reverse=True):
                    list_bs.pop(idx)
                    list_pop_idt.pop(idx)

            # Configure columns heads for saved csv
            columns = ["count"]
            for b_idx in range(nb):
                columns.extend(["ibl", "ca", "bl"])

            # Save to csv
            path_bs_save = dir_bs_save.joinpath(f"{idt}.csv")
            df = pd.DataFrame(list_bs, index=list_pop_idt, columns=columns)
            df.to_csv(path_bs_save)

    @property
    def bt_format(self):
        """    Bond type format dict    """
        return {'Unknown': "?", 'Single': "-", 'Double': "=", 'Triple': "#", 'Quadruple': "$",
                'Aromatic': "@", 'Delocalised': "--", 'Pi': "&"}

    def clear(self):
        """    Clear all of data stored in self    """
        self.list_target = []  # To store CSD instance, Entry, Crystal, Molecule
        self.searchers = {}  # To store searchers
        self.dict_bs = {}  # To store bonds' structure for pairs with a specific graph information
        self.atomic_radius = None  # Covalent radius

    def config_atom_radius_info(self, path_excel: Path):
        """    Read atomic radius information from excel to configure self.atomic_radius    """
        df = pd.read_excel(path_excel, engine="openpyxl", index_col=0)
        self.atomic_radius = {i: df.loc[i, "radius"] for i in df.index}

    def config_substructure_searchers(self, *mols: Molecule, files_dir: Path = None,
                                      dict_smarts: Dict = None):
        """
        Config substructure searchers
        :param mols: molecules (mols) used to defining searchers
        :param files_dir: dir of files.con used to defining searchers
        :param dict_smarts: dict of SMILES string used to defining searchers
        :return: None
        """
        # Define substructures
        dict_sst = {}  # list of substructures (sst)
        if mols:
            dict_sst.update({m.identifier: search.MoleculeSubstructure(m) for m in mols})
        if files_dir:
            for con_file in files_dir.glob("*"):
                if con_file.is_file() and con_file.suffix == ".con":
                    dict_sst[con_file.stem] = search.ConnserSubstructure(str(con_file))
                else:  # for sst defined by multi-files
                    dict_sst[con_file.stem] = [search.ConnserSubstructure(str(csf))
                                               for csf in con_file.glob("*.con")]
        if dict_smarts:
            dict_sst.update({name: search.SMARTSSubstructure(smarts)
                             for name, smarts in dict_smarts.items()})

        # Config searchers
        for name, sst in dict_sst.items():
            if isinstance(sst, list):
                list_sst = []
                for a_sst in sst:
                    searcher = search.SubstructureSearch()
                    searcher.add_substructure(a_sst)
                    list_sst.append(searcher)
                self.searchers[name] = list_sst
            else:
                searcher = search.SubstructureSearch()
                searcher.add_substructure(sst)
                self.searchers[name] = searcher

        self.searchers = {n: self.searchers[n] for n in sorted(self.searchers, reverse=True)}

    @staticmethod
    def classify_scp_graph_structure(list_scp, record_same_graph_pairs=True):
        """
        Classify single connection pairs according to the graph structure.
        :param list_scp: list of single connected pairs, that is generated from same natural pair by
                         self.generate_single_connected_pairs()
        :param record_same_graph_pairs:
        :return:
        """

        dict_scp_graph_type = {}
        # For a single connected pair (scp) to find out other scp with same graph structure
        while list_scp:
            main_scp = list_scp.pop()
            list_unique_scp = [main_scp]
            # Define a graph structure searcher
            similarity_searcher = search.SimilaritySearch(main_scp)
            list_pi = []  # list of index to pop
            # Compare the similarity between the main_scp and others in the list one by one
            for idx, other_scp in enumerate(list_scp):
                # Calculating similarity between this pair and comparison pair
                search_result = similarity_searcher.search_molecule(other_scp)
                if search_result.similarity == 1:
                    list_pi.append(idx)  # Recording the index of pairs in the list_scp

            # Pop the corresponding ligand in reverse order through the recorded index
            for pi in sorted(list_pi, reverse=True):
                same_pair = list_scp.pop(pi)
                # If need to record the pairs, which have same graph structure with the main_scp
                if record_same_graph_pairs:
                    list_unique_scp.append(same_pair)

            # Write the list_unique_scp into dict_scp_graph_type
            count = 0
            # coordination atom symbol in this main pair
            cas = [a for a in main_scp.atoms if a.is_metal][0].neighbours[0].atomic_symbol
            while f"{cas}{count}" in dict_scp_graph_type:
                count += 1
            dict_scp_graph_type[f"{cas}{count}"] = list_unique_scp

        return dict_scp_graph_type

    def classify_pairs_by_ligand(self):
        """"""
        dict_ligand = {}
        for pair in tqdm(self.list_target):
            if not isinstance(pair, Molecule):
                raise TypeError("the pair should be a Molecule!")
            pair.remove_atoms(a for a in pair.atoms if a.is_metal)
            try:
                pair.assign_bond_types()
                pair.add_hydrogens()
            except:
                pass
            list_idt = dict_ligand.setdefault(pair.smiles, [])
            list_idt.append(pair.identifier)

        return dict_ligand

    # TODO: Rewrite
    def eliminate_redundant_mols(self, *mols: Container[Molecule]):
        """"""
        will_return = True if mols else False
        mols = mols if mols else self.list_target
        dict_mols = {}
        dict_smiles = {}  # Record smiles types for mols with exactly graph
        for mol in tqdm(mols, "Classify mols by number of atoms", colour="#00BBBB", leave=False):
            list_mols = dict_mols.setdefault(len(mol.atoms), [])
            list_mols.append(mol)

        total = len(mols)
        count = 0
        mols = []
        for list_mols in dict_mols.values():
            while list_mols:
                mol = list_mols.pop()
                mols.append(mol)
                # Config similarity searcher
                similarity_searcher = search.SimilaritySearch(mol)
                list_smiles = dict_smiles.setdefault(mol.smiles, [])
                list_smiles.append(mol.smiles)
                removed_idx = []
                for idx, other_mol in enumerate(list_mols):
                    search_result = similarity_searcher.search_molecule(other_mol)
                    if search_result.similarity == 1:
                        removed_idx.append(idx)
                        if other_mol.smiles not in list_smiles:
                            list_smiles.append(other_mol.smiles)

                count += len(removed_idx) + 1
                for idx in sorted(removed_idx, reverse=True):
                    list_mols.pop(idx)
                print(f"\r{count}/{total}", end="")

        if will_return:
            return dict_smiles, mols
        else:
            self.list_target = mols
            return dict_smiles

    @staticmethod
    def file_reader(
            file_dir: Path, reader: str = "Molecule",
            file_type: str = "cif",
            generator=False,
            ranges: Container = None,
            condition: Callable = None
    ) -> Union[List[Union[Molecule, Crystal, Entry]], Generator]:
        """
        Read data from disk
        :param condition:
        :param ranges:
        :param generator: if True, it will return a generator.
        :param file_dir: the dir to storing read files
        :param reader:
        :param file_type: read file type
        :return: List of bond_length target from files
        """
        # Check arguments
        if reader not in ["Entry", "Crystal", "Molecule"]:
            raise ValueError(f"The {reader} writer is not supported!")

        files_reader = eval(f"io.{reader}Reader")  # Defining the type of files reader
        list_file = file_dir.glob("*") if file_type == "all" else file_dir.glob(f"*.{file_type}")

        if generator:
            return (
                files_reader(str(f))[0]
                for i, f in enumerate(list_file)
                if (not ranges or i in ranges) and (condition is None or condition(f))
            )
        else:
            return [
                files_reader(str(f))[0]
                for i, f in tqdm(enumerate(list_file), f"{reader}.{file_type} reading", colour="##BBBB00", leave=False)
                if (not ranges or i in ranges) and (condition is None or condition(f))
            ]

    def file_writer(self, save_dir: Path, *files, writer: str = "Molecule", file_type: str = "cif",
                    overwrite=True):
        """
        Save data to disk
        :param overwrite: If true allowed to overwrite files when its name same, else write by name with count
        :param save_dir: Saving dir
        :param files: Files needed to save
        :param writer:
        :param file_type: saved file type, ".sdf" is recommended for Entry; ".mol2" is recommended for Molecules
        :return:
        """
        # Check arguments
        if writer not in ["Entry", "Crystal", "Molecule"]:
            raise ValueError(f"The {writer} writer is not supported!")
        # Ensure save dir is exist
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        files_writer = eval(f"io.{writer}Writer")  # Defining the type of files writer
        files = files if files else self.list_target

        # Make Sure save_dir is exist
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True)

        for file in tqdm(files, f"{writer}.{file_type} writing", colour="blue", leave=False):

            path_file = save_dir.joinpath(f"{file.identifier}.{file_type}")
            count = 0
            while path_file.exists() and not overwrite:
                path_file = save_dir.joinpath(f"{file.identifier}-{count}.{file_type}")
                count += 1

            with files_writer(path_file) as w:
                w.write(file)

    def filter_r_factor(self, *entries, min_factor=None, max_factor=None, allow_none_r_factor=False):
        """
        filter out entries out of allowed scope
        :param allow_none_r_factor: whether the r factors are allowed to be None
        :param entries: input entries
        :param min_factor: lowest bound of allowed scope
        :param max_factor: highest bound of
        :return:
        """
        will_return = True if entries else False
        entries = entries if entries else self.list_target

        list_entry = []
        for entry in entries:
            # Check target's type
            if not isinstance(entry, Entry):
                raise TypeError(f"only Entry class is supported, but get a {type(entry)}")

            # Filtering
            r_factor = entry.r_factor
            if (
                    (allow_none_r_factor or r_factor)
                    and ((not min_factor and min_factor != 0) or r_factor >= min_factor)
                    and ((not max_factor and max_factor != 0) or r_factor <= max_factor)
            ):
                list_entry.append(entry)

            if will_return:
                return list_entry
            else:
                self.list_target = list_entry

    def filter_targets(self, method: Callable):
        """

        :param method: Callable objects, if a targets could pass through the filter, return True
        :return:
        """
        return [t for t in tqdm(self.list_target, "filter targets", leave=False) if method(t)]

    def filter_organic(self, reverse=False):
        """
        Filter out inorganic molecules
        :param reverse: Filter out organic molecules instead
        :return:
        """
        self.list_target = [
            t
            for t in tqdm(
                self.list_target, f"filter out {'organic' if reverse else 'inorganic'} mol",
                colour="#BBBB00", leave=False
            )
            if (isinstance(t, Molecule) and t.is_organic) or
               (isinstance(t, (Entry, Crystal)) and t.molecule.is_organic)
        ]

        if not self.list_target:
            raise ValueError(f"None of {'inorganic' if reverse else 'organic'} mols are residue!")

    def filter_mol_with_isolated_atom(self):
        """    Filter out mol, crystal, entry, with isolated atom    """
        self.list_target = [
            t
            for t in tqdm(
                self.list_target, f"filter out mol with isolated atom",
                colour="#BBBB00", leave=False
            )
            if (isinstance(t, Molecule) and len(t.atoms) > 1) or
               (isinstance(t, (Crystal, Entry)) and len(t.molecule.atoms) > 1)
        ]

        # Raise error when nothing left
        if not self.list_target:
            raise ValueError(f"None of targets are residue!")

    def generate_diagram(self, dir_diagram_save: Path, *targets, file_type="png"):
        """
        Generate 2D diagram.
        :param file_type: file type to save
        :param dir_diagram_save: the dir to save diagram
        :param targets: [option] if not given, get target from self.list_target
        :return:
        """
        # Determining data source
        targets = targets if targets else self.list_target

        # Make sure save diagram is exist
        if not dir_diagram_save.exists():
            dir_diagram_save.mkdir()

        # Initialize diagram generate
        dg = DiagramGenerator()

        for target in tqdm(targets, "generate diagram", colour="#00BBBB", leave=False):
            idt = target.identifier
            img = dg.image(target)
            img.save(dir_diagram_save.joinpath(f"{idt}.{file_type}"))

    def generate_mol_conformer(self, *mols, max_conformers=1, return_first_mol=True):
        """
        Generate conformer for list of mols
        :param max_conformers:
        :param return_first_mol: Whether to return the mol object with first conformers instead
        :return: list of conformers
        """
        cg = ConformerGenerator()  # conformer_generator
        cg.settings.max_conformers = max_conformers

        mols = mols if mols else self.list_target
        if return_first_mol:
            return [cg.generate(m)[0].molecule for m in tqdm(mols, "generate conformer", colour="#BB00BB", leave=False)]
        else:
            return [cg.generate(m) for m in tqdm(mols, "generate conformer", colour="#BB00BB", leave=False)]

    @staticmethod
    def generate_mol_from_smiles(list_smiles, list_id=None):
        """
        Generate Molecule objects from corresponding smiles.
        :param list_id: list of identifier for corresponding smiles, if not given, the identifier of mol will be smiles
        :param list_smiles: input list_smiles.
        :return: list of Molecule objects.
        """
        # Check args
        if list_id and len(list_smiles) != len(list_id):
            raise ValueError("the args list_smiles and list_id should be same length")

        list_mol = []
        for idx, smiles in enumerate(tqdm(list_smiles, "3d mol from smiles:", colour="#BB00BB", leave=False)):
            try:
                mol = Molecule.from_string(smiles)
                mol.identifier = list_id[idx]
                list_mol.append(mol)
            except:
                print(smiles)

        return list_mol

    @staticmethod
    def generate_single_connected_pairs(pair):
        """
        Generate single connected metal-ligand pairs from natural metal-ligand pair.
        In this function, any pairs whose metal and ligand connected by multiple coordination bonds will generate/
        its variants, which leave only one coordination bond and the rest of will be broken.
        :param pair: the input natural pair
        :return:
        """
        # Generating single connected pairs (scp)
        # metal label in a pair
        ml = [a.label for a in pair.atoms if a.is_metal][0]
        # list of coordination atoms' labels
        list_cal = [a.label for a in pair.atom(ml).neighbours]
        if len(list_cal) > 1:
            list_scp = [pair.copy() for cal in list_cal]  # list of single connected pairs (csp)
            # Remove coordination bonds from ith pair except for ith bond
            for idx, csp in enumerate(list_scp):
                csp.remove_bonds(csp.bond(ml, cal) for i, cal in enumerate(list_cal) if i != idx)
        else:
            list_scp = [pair]

        return list_scp

    def generate_virtual_pairs(
            self, list_cm_symbols: List[str],
            *ligands: List[Molecule],
            selected_fg: List[str] = None,
            bond_type="Single",
            sorted_ca: List[List[str]] = None
    ):
        """
        Generate virtual Metal-Ligand Pairs by link the ligand with coordination metals
        :param list_cm_symbols: list of metal's symbols
        :param ligands: the list of input ligands. if not given, the list of ligands will come from self.list_target,
                        meanwhile the generated pairs will replace the original objects stored in self.list_target.
        :param selected_fg: list of names for function groups to be seen as the coordination function groups to link to
                            coordination metals
        :param bond_type: the bond type of coordination bonds, it must be choose from "single", "double", "Triple",
                          "Quadruple", "Aromatic" or "Delocalised".
        :param sorted_ca: the coordination atoms in coordination function groups and their priority
                          to be considered to link with coordination metals
        :return: the generated pairs, or None if the arg: ligands is not given.
        """
        sorted_ca_symbols = sorted_ca if sorted_ca else [["O"], ["N"], ["S"], ["P"]]
        will_return = True if ligands else False
        ligands = ligands if ligands else self.list_target

        # Check weather the substructure searchers are configured.
        if not self.searchers:
            raise AttributeError("the substructures searchers are not configured!")

        # Check weather the selected searchers are in the configured searchers.
        if selected_fg:
            for fg_name in selected_fg:
                if fg_name not in self.searchers:
                    raise KeyError(f"There is not substructure searchers named {fg_name} configured!")

        # Manufacturing coordination metals
        list_cm = [Atom(s, label=f"{s}1") for s in list_cm_symbols]

        list_virtual_pairs = []
        for ligand in tqdm(ligands, "Generate virtual pairs", colour="#00BBBB", leave=False):  # For each ligand
            # Search substructures
            set_hit_atom_labels = set()  # To recording atoms which have been hit to avoid to hit it again.
            ca_labels_link = []  # To recording labels of coordination atoms to link to coordination metals
            for name, searcher in self.searchers.items():
                hits = [h for s in searcher for h in s.search(ligand)] \
                    if isinstance(searcher, List) else searcher.search(ligand)
                # Filter out those substructures which have been hit before.
                hits = [h for h in hits if all(a.label not in set_hit_atom_labels for a in h.match_atoms())]
                set_hit_atom_labels.update([a.label for h in hits for a in h.match_atoms() if a.atomic_symbol != "C"])
                # If searched substructure in the ligand meanwhile the substructure is selected or not given.
                if hits and (not selected_fg or (selected_fg and name in selected_fg)):
                    hit_atom_symbols = {a.atomic_symbol for h in hits for a in h.match_atoms()}
                    for ca_symbols in sorted_ca_symbols:
                        if any(s in ca_symbols for s in hit_atom_symbols):
                            ca_labels_link.extend(
                                [a.label for h in hits for a in h.match_atoms() if a.atomic_symbol in ca_symbols]
                            )
                        break

            # For each recorded coordination atoms used to link
            for ca_label in ca_labels_link:
                for cm in list_cm:
                    virtual_pair = ligand.copy()
                    ca = virtual_pair.atom(ca_label)
                    ca_coordinate = np.matrix(ca.coordinates)
                    bond_vector = np.matrix([0.0, 0.0, 0.0])
                    for neighbour_atom in ca.neighbours:
                        bond_vector += ca_coordinate - np.matrix([neighbour_atom.coordinates])
                    i_vector = bond_vector / pow((bond_vector * bond_vector.T).max(), 1 / 2)

                    cm = virtual_pair.add_atom(cm)
                    virtual_pair.add_bond(bond_type, cm, ca)
                    bond = virtual_pair.bond(cm.label, ca.label)
                    ibl = bond.ideal_bond_length
                    cm.coordinates = list((ibl * i_vector + ca_coordinate).A1)
                    virtual_pair.identifier = virtual_pair.identifier + f"-{cm.atomic_symbol}"
                    list_virtual_pairs.append(virtual_pair)

        if will_return:
            return list_virtual_pairs
        else:
            self.list_target = list_virtual_pairs

    @staticmethod
    def get_bond_length_median(
            dir_raw_bs: Path,
            dir_cooked_bs: Path,
            condition: Callable = None
    ) -> List:
        """
        Read bond structure information, which got from different pairs with same graph structure,\
        from files.csv in dir_raw_bs.
        Because of the same graph in same file.csv, these bond structure infos have same exactly metal and\
        coordination atoms, while the coordination bond length might be different.
        Merging all of bond structure infos to one, in which the bond length items are the median of all\
        corresponding bonds' length in each pairs.
        :param condition: <Callable> Return True when a file is needed to cook, else False.
        :param dir_raw_bs: The directory that raw bond structure infos come from.
        :param dir_cooked_bs: The directory to save cooked bond structure infos
        :return:
        """
        # Make sure the save directory is exist!
        if not dir_cooked_bs.exists():
            dir_cooked_bs.mkdir(parents=True)

        null_files = []
        for path_bs_csv in tqdm(dir_raw_bs.glob("*.csv"), "cook_bs_info", colour="#29B59E", leave=False):

            # Pass when condition return a False
            if condition is not None and not condition(path_bs_csv):
                continue

            df = pd.read_csv(path_bs_csv, index_col=0)
            try:
                save_array = df.iloc[0, :].values
            except IndexError:
                null_files.append(path_bs_csv.name)
                print(path_bs_csv.name)
                continue

            nca = int((df.shape[1] - 1) / 3)
            for ca_idx in range(nca):
                bls = df.iloc[:, 3 + 3 * ca_idx]
                bl_median = bls.median()
                save_array[3 + 3 * ca_idx] = bl_median

            name = path_bs_csv.name
            try:
                save_df = pd.DataFrame(save_array, columns=[path_bs_csv.stem], index=df.columns)
            except ValueError:
                print(path_bs_csv)
                raise
            save_df = save_df.T
            save_df.to_csv(dir_cooked_bs.joinpath(name))

        return null_files

    def get_bond_structure_info(self, dir_bond_structure: Path):
        """    TODO    """
        dict_bs = {}
        for pair in self.list_target:
            list_bs = dict_bs.setdefault(pair.identifier, [])
            metal = [a for a in pair.atoms if a.is_metal][0]
            cas = metal.neighbours
            bs = [len(cas)]
            for ca in cas:
                bond = pair.bond(metal.label, ca.label)
                ibl = bond.ideal_bond_length
                bl = bond.length
                bs.extend([ibl, ca.atomic_symbol, bl])
            list_bs.append(bs)

        # Make sure save directory is exist.
        if not dir_bond_structure.exists():
            dir_bond_structure.mkdir(parents=True)

        # To write
        for idt, list_bs in dict_bs.items():
            path_bs = dir_bond_structure.joinpath(f"{idt}.csv")

            columns = ["nca"]
            for i in range(list_bs[0][0]):
                columns.extend(["ibl", "ca", "bl"])

            df = pd.DataFrame(list_bs, columns=columns)
            df.to_csv(path_bs)

    def get_cdn_bs_from_scp(self, pair):
        """    Get coordination (cdn) bond structures (bs) from single connected pairs    """
        # Get the metal in the pair
        list_mtl = [a for a in pair.atoms if a.is_metal]
        if len(list_mtl) != 1:
            raise AssertionError("The number of metal in the pair do not be one!!")
        mtl = list_mtl[0]
        # Get metal's label and symbol
        ml, ms = mtl.label, mtl.atomic_symbol
        # Generate of list of single connected pairs from the natural pair
        list_scp = self.generate_single_connected_pairs(pair)
        # Initialization of bond structure record list
        if len(list_scp) == len(mtl.neighbours):
            list_bs = [len(list_scp)]
        else:
            raise AssertionError(f"Failure to generate single connected pairs (scp), get a error number of scp,"
                                 f"which were expected to be {len(mtl.neighbours)}, but got {len(list_scp)}")
        dict_scp_graph_type = self.classify_scp_graph_structure(list_scp)

        # Get bond structure info by name's order of coordination atoms type
        for cat in sorted(dict_scp_graph_type):
            # list of scp with same graph (sg)
            list_sg_scp = dict_scp_graph_type[cat]
            for sg_scp in list_sg_scp:
                scp_metal = sg_scp.atom(ml)
                cbs = scp_metal.bonds
                if len(cbs) != 1:
                    raise AssertionError("the pair do not a single connected pair!")
                cb = cbs[0]
                try:
                    bl = cb.length
                    ibl = cb.ideal_bond_length
                    list_bs.extend([ibl, cat, bl])

                except BaseException:

                    return False

        return list_bs

    def get_entry(self, list_identifier: List[str] = None):
        """    Get list of Entries by a given identifier    """
        entry_reader = io.EntryReader("CSD")
        if list_identifier is None:
            self.list_target = entry_reader
        else:
            list_entry = []  # List to store get entries
            list_except = []
            for idt in tqdm(list_identifier, "Loading entry", colour="blue", leave=False):
                try:
                    list_entry.append(entry_reader.entry(idt))
                except:
                    list_except.append(idt)

            self.list_target = list_entry

            if list_except:
                print(f"\r{list_except} are not loaded!")

    @classmethod
    def get_identifier_for_each_metals(cls, save_path: Union[Path, str], *metals: str):
        """
        Get identifiers of specific metals
        :param save_path: the path of excel to save identifiers
        :param metals: The target metals to get identifiers
        :return:
        """
        all_metal = True if "metal" in metals else False
        mol_reader = io.MoleculeReader("CSD")
        dict_identifier = {}
        for mol in tqdm(mol_reader, "Getting identifier", colour="#00BBBB", leave=False):
            mol_id = mol.identifier
            if all_metal:
                set_metal_symbol = set(a.atomic_symbol for a in mol.atoms if a.is_metal)
            else:
                set_metal_symbol = set(a.atomic_symbol for a in mol.atoms if a.atomic_symbol in metals)
            for mtl in set_metal_symbol:
                list_id = dict_identifier.setdefault(mtl, [])
                list_id.append(mol_id)

        # Make sure the excel_dir is exist
        save_dir = save_path.parent
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True)

        # Save to Excel
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for mtl, list_identifier in dict_identifier.items():
                ser = pd.Series(list_identifier, name="identifier")
                ser.to_excel(writer, sheet_name=mtl, index=False)

    def get_ligands(self, *targets, n_bar=False):
        """    Get ligands info from Molecule, Crystal or Entry    """
        will_return = True if targets else False
        targets = targets if targets else self.list_target
        bar = nothing_bar if n_bar else tqdm

        targets = self.transform_crystal_to_molecule(*targets, n_bar=n_bar)
        list_mols = []
        for mol in tqdm(targets, "Get ligands", colour="#00BBBB", leave=False):
            mol.remove_atoms(a for a in mol.atoms if a.is_metal)
            idt = mol.identifier
            for ligand_count, comp in enumerate(mol.components):
                if not mol.is_organic:
                    continue
                comp.assign_bond_types()
                comp.assign_partial_charges()
                try:
                    comp.add_hydrogens()
                except:
                    pass

                comp.identifier = f"{idt}-{ligand_count}"
                list_mols.append(comp)

        if will_return:
            return list_mols
        self.list_target = list_mols

    def get_mol_from_container(self):
        """"""
        self.list_target = [c.mol for c in self.list_target]

    def get_pairs_bs(self, *pairs, n_bar=True):
        """    Get bond structure from pairs    """
        # Determining data source
        will_return = True if pairs else False
        pairs = pairs if pairs else self.list_target
        dict_bs = {}
        bar = nothing_bar if n_bar else tqdm
        for pair in bar(pairs, "Getting pairs bs", colour="#00BBBB", leave=False):
            # the values of dict is a 2D list
            dict_bs[pair.identifier] = [[pair.identifier], [self.get_cdn_bs_from_scp(pair)]]

        if will_return:
            return dict_bs
        else:
            self.dict_bs = dict_bs

    @classmethod
    def graph_files_arrangement(cls, dir_graph: Union[str, Path]):
        """"""
        # Argument check
        if isinstance(dir_graph, str):
            dir_graph = Path(dir_graph)
        elif not isinstance(dir_graph, Path):
            raise TypeError(f"the arg dir_graph only allow str or Path class, but get a {type(dir_graph)} instead")

        df_bs = pd.read_csv(dir_graph.joinpath("bs.csv"), index_col=0)
        set_bsi = set(df_bs.index)  # set of bond structure identifier (bsi)
        set_its = set_bsi.copy()  # set of intersection (its)

        # list of set of graph data identifier
        list_gd_dir = [d for d in dir_graph.glob("*") if d.is_dir()]  # list of graph data (gd) dir
        list_set_idt = [{p.stem for p in d.glob("*")} for d in list_gd_dir]

        # Calculating intersection (its) of all identifier set
        for set_idt in list_set_idt:
            set_its.intersection_update(set_idt)

        # Calculating difference between each identifier set and the intersection set
        list_set_idt.append(set_bsi)
        for set_idt in list_set_idt:
            set_idt.difference_update(set_its)

        # Remove different csv file from graph data dir
        for idx, gd_dir in enumerate(list_gd_dir):
            for csv_stem in list_set_idt[idx]:
                os.remove(gd_dir.joinpath(f"{csv_stem}.csv"))

            print(f"\r{len(list_set_idt[idx])} csv files are removed from {gd_dir.name} dir!")

        # Remove items with identifier in difference bs set
        df_bs.drop(labels=list_set_idt[3], inplace=True)
        print(f"\r{len(list_set_idt[3])} csv files are removed from bs.csv")
        # Overwrite bs.csv
        df_bs.to_csv(dir_graph.joinpath("bs.csv"))

    @staticmethod
    def is_solvent(c, solvent_smiles):
        """    Check whether a components is a solvent    """
        return c.smiles == 'O' or c.smiles in solvent_smiles

    @staticmethod
    def is_multidentate(c, mol, dentate_number=2):
        """
        Check for components bonded to metals more than once.
        :param c: a Component (c) in the molecule (mol)
        :param mol: a Molecule (mol) contains the component
        :param dentate_number: the minimum Number that a multidentate component bond to metals
        :return: bool
        """
        got_num = 0
        for a in c.atoms:
            orig_a = mol.atom(a.label)
            if any(a.is_metal for a in orig_a.neighbours):
                if got_num >= dentate_number:
                    return True
                got_num += 1
        return False

    def match_pair_and_bs(self, list_pairs=None, dict_bs=None):
        """
        Make sure each pair graph data have their bond structure data, and vice versa
        :param list_pairs: input pair graph data
        :param dict_bs: input bond structure data
        :return:
        """
        # Setting processing object
        will_return = False if (list_pairs is None or dict_bs is None) else True
        list_pairs = self.list_target if list_pairs is None else list_pairs
        dict_bs = self.dict_bs if dict_bs is None else dict_bs

        dict_pairs = {p.identifier: p for p in list_pairs}
        list_discard = []
        while len(dict_pairs) != len(dict_bs):
            is_pairs_long = True if len(dict_pairs) > len(dict_bs) else False
            short_dict = dict_bs if is_pairs_long else dict_pairs
            long_dict = dict_pairs if is_pairs_long else dict_bs

            short_temp_dict = {}
            long_temp_dict = {}
            short_length = len(short_dict)

            while short_dict:

                # Pop any item from the shorter dict
                idt, sd = short_dict.popitem()
                # git item from longer dict by idt if it's existing, else get a False
                ld = long_dict.get(idt, False)

                if ld:  # if item is existing both in shorter and longer, recording it else skipping
                    short_temp_dict[idt] = sd
                    long_temp_dict[idt] = ld
                    _ = long_dict.pop(idt)
                else:
                    list_discard.append(idt)

                # Precess bar
                p_bar(short_length - len(dict_pairs), short_length, "marching:", 25)

            dict_pairs = long_temp_dict if is_pairs_long else short_temp_dict
            dict_bs = short_temp_dict if is_pairs_long else long_temp_dict

        # Reporting if discard operating is carried out
        if list_discard:
            print(f"Discard items {len(list_discard)}")

        if will_return:
            return list(dict_pairs.values()), dict_bs
        else:
            self.list_target, self.dict_bs = list(dict_pairs.values()), dict_bs

    def molecular_graph_transform(self, save_dir: Path, *mols, outermost_layer=True):
        """
        Transform molecules to their graph representations (gr).
        And then, save them as a excel file
        :param save_dir: the excel save dir
        :param mols: input molecules, if not give read from self.list_target
        :param outermost_layer: Whether get outermost open shell electron's structure, or instead of
                                getting all of electron's structure, as a atom's orbital feature.
        :return: None
        """
        # Basic parameters
        tuple_excel_name = ("adjacency", "atomic_feature", "bond_feature")
        bond_feature_name = ("bond_type", "is_conjugated", "is_rotatable")

        # Ensure all directions are exist.
        for dirs in tuple_excel_name:
            the_dir = save_dir.joinpath(dirs)
            if not the_dir.exists():
                the_dir.mkdir(parents=True)

        mols = mols if mols else self.list_target
        dict_graph_representation = {}  # Results repository
        for idx, mol in enumerate(tqdm(mols, "\rgraph transform", colour="blue", leave=False)):

            # Calculating atoms' labels in the molecule (mol_atoms_lbl)
            # Calculating molecular graph adjacency matrix (adjacency)
            # Calculating molecular feature matrix
            mol_atoms_lbl, adjacency, atomic_feature_matrix = \
                self._molecular_graph_adjacency_and_feature_matrix(mol, outermost_layer)

            # Calculating bond features
            bond_feature_matrix, bond_name = \
                self._bond_feature_matrix(mol, mol_atoms_lbl, adjacency, bond_feature_name)

            # transform to a DataFrame
            adjacency = pd.DataFrame(adjacency, columns=mol_atoms_lbl, index=mol_atoms_lbl)
            if len(atomic_feature_matrix) > 1:  # if atoms are more than 1
                atomic_feature_matrix = pd.DataFrame(atomic_feature_matrix)
            else:
                atomic_feature_matrix = pd.DataFrame(
                    list(atomic_feature_matrix.values()), index=list(atomic_feature_matrix.keys())
                ).T
            bond_feature_matrix = pd.DataFrame(bond_feature_matrix, columns=bond_feature_name, index=bond_name)

            # Save graph results to csv
            adjacency.to_csv(save_dir.joinpath(f"adjacency/{mol.identifier}.csv"))
            atomic_feature_matrix.to_csv(save_dir.joinpath(f"atomic_feature/{mol.identifier}.csv"))
            bond_feature_matrix.to_csv(save_dir.joinpath(f"bond_feature/{mol.identifier}.csv"))

    @staticmethod
    def molecular_standardization(*mols, normalise_label=False, remove_unknown_atoms=False, add_hydrogens=False,
                                  remove_hydrogens=False, normalise_hydrogens=False, normalise_atom_order=False,
                                  assign_bond_types=False, assign_partial_charges=False, n_bar=True, **kwargs):
        """
        Normalising molecules
        Detail in web: https://downloads.ccdc.cam.ac.uk/documentation/API/index.html
        """
        if any(not isinstance(m, Molecule) for m in mols):
            raise TypeError("The arg mols only support Molecule class")
        if add_hydrogens and remove_hydrogens:
            raise ValueError("The args of 'add_hydrogens' and 'remove_hydrogens' aren't allowed to be both True!")

        # Standardization of molecules
        fail_assign_bond_type = []
        fail_normalise_atom_order = []
        return_mols = []
        bar = nothing_bar if n_bar else tqdm
        for mol in bar(mols, "Standardization of molecule", colour="blue", leave=False):

            if normalise_label:
                mol.normalise_labels()
            if remove_unknown_atoms:
                mol.remove_unknown_atoms()
            if assign_bond_types:
                if mol.all_atoms_have_sites:
                    mol.assign_bond_types(**kwargs)
                else:
                    fail_assign_bond_type.append(mol.identifier)
            if remove_hydrogens:
                mol.remove_hydrogens()
            if add_hydrogens:
                mol.add_hydrogens()
            if normalise_hydrogens:
                mol.normalise_hydrogens()
            if normalise_atom_order:
                try:
                    mol.normalise_atom_order()
                except:
                    fail_normalise_atom_order.append(mol.identifier)
            if assign_partial_charges:
                mol.assign_partial_charges()

            return_mols.append(mol)

        # Report unexpected
        if fail_assign_bond_type and any(idt.split("-")[1] in ["AWOVEO", "FAGVEO02", "IJOPIH"]
                                         for idt in fail_assign_bond_type):
            print(f"fail_assign_bond_type, number: {len(fail_assign_bond_type)} {fail_assign_bond_type}\n")
        if fail_normalise_atom_order:
            print(f"fail_normalise_atom_order, number: {len(fail_normalise_atom_order)} {fail_assign_bond_type}")

        return return_mols

    def optimize_configuration(self, *mols):
        """    Optimize molecules configuration    """
        # Determining data source
        will_return = True if mols else False
        mols = mols if mols else self.list_target

        # Configure conformer of molecular minimiser
        cmm = MoleculeMinimiser()

        minimized_mols = []
        for mol in tqdm(mols, "Optimize Configuration", colour="#00BBBB", leave=False):
            minimized_mols.append(cmm.minimise(mol))

        if will_return:
            return minimized_mols
        self.list_target = minimized_mols

    def read_ideal_bond_length_from_csv(self, path_csv, bond_type=None):
        """    Read the data of ideal bond length and return them by a DataFrame    """
        ser = pd.read_csv(path_csv, index_col=0)
        return {k: v for k, v in ser.items() if not bond_type or self.bt_format[bond_type] in k}

    @staticmethod
    def read_identifier_from_excel(path_excel: Union[Path, str], *elements):
        """
        Read identifier from the excel writer by the function, self.get_identifier_for_each_metals().
        :param path_excel: Path of excel
        :param elements: elements need to get identifiers
        :return: list or dict of identifiers
        """
        excel_dir = path_excel.parent
        if not excel_dir:
            excel_dir.mkdir(exist_ok=True)
        excel = pd.ExcelFile(path_excel, engine="openpyxl")
        dict_identifier = {el: excel.parse(el).values.flatten().tolist()
                           for el in excel.sheet_names
                           }

        if not elements:
            return dict_identifier
        if len(elements) == 1:
            return dict_identifier[elements[0]]
        else:
            return {el: idt
                    for el, idt in dict_identifier.items()
                    if el in elements
                    }

    def remove_isolated_atom_from_entry(self, *entries):
        """    Remove isolated atoms in a Entry    """
        will_return = True if entries else False
        entries = entries if entries else self.list_target
        removed_entries = []
        for entry in tqdm(entries, "remove isolated atoms", colour="blue", leave=False):
            mol = entry.molecule
            mol.normalise_labels()
            to_remove = [c for c in mol.components if len(c.atoms) == 1]
            if to_remove:
                mol.remove_atoms(mol.atom(c.atoms[0].label) for c in to_remove)

            # if molecule have any atoms
            if mol.atoms:
                entry.crystal.molecule = mol
                removed_entries.append(entry)

        if will_return:
            return removed_entries
        else:
            self.list_target = removed_entries

    def remove_solvents_and_anion_from_entries(self, *entries, solvents: Union[Molecule, str] = None, mode="append",
                                               remove_multidentate=True, mol_file_dir: Path = None):
        """
        Delete solvents from crystals.
        There is 74 default 74 solvents by CSD
        :param mol_file_dir: the dir of mol files need to remove from a entry
        :param remove_multidentate: Whether to retain multidentate components
        :param solvents: Custom solvent molecules, given by smiles or Molecules
        :param mode: Whether "append" or "inplace" custom solvents to default 74 solvents
        :return: list entry if entries are given else None
        """
        # Check args
        if mode not in ["append", "inplace"]:
            raise ValueError(f"the arg mode should be 'append' or 'inplace', but get {mode}")
        if not solvents and mode == "inplace":
            raise ValueError("the arg mode shouldn't be 'inplace' when custom solvents not be given!")

        # Load smiles of solvents
        # Load smiles of custom solvents
        solvents_smiles = []
        if solvents:
            for solvent in solvents:
                if isinstance(solvent, Molecule):
                    solvents_smiles.append(solvent.smiles)
                elif isinstance(solvent, str):
                    solvents_smiles.append(solvent)
                else:
                    TypeError(f"{type(solvent)} is not supported by arg solvents!")

        # Load smiles of other mol files
        if mol_file_dir and mol_file_dir.exists():
            solvents_smiles.extend(
                [io.MoleculeReader(str(f))[0].smiles
                 for f in mol_file_dir.glob("*")
                 ]
            )
        elif mol_file_dir:
            raise FileExistsError(f"The dir {mol_file_dir} is not exist!")

        # Load smiles of default solvents
        if not solvents or mode == "append":
            if sys.platform == "linux":
                solvents_dir = Path(io.csd_directory()).joinpath("../molecular_libraries/ccdc_solvents")
            elif sys.platform in ("win32", "win64"):
                solvents_dir = Path(io.csd_directory()).joinpath("../Mercury/molecular_libraries/ccdc_solvents")
            else:
                raise EnvironmentError(
                    f"The supported operating systems only include Linux, Win32 and win64, not {sys.platform}"
                )

            if solvents_dir.exists():
                solvents_smiles.extend(
                    [io.MoleculeReader(str(f))[0].smiles
                     for f in solvents_dir.glob("*.mol2")
                     ]
                )
            else:
                raise FileExistsError(f"The dir {solvents_dir} is not exist!")

        # Remove solvents from entry
        list_entry = []
        will_return = True if entries else False
        entries = entries if entries else self.list_target
        for entry in tqdm(entries, "solvents removing", colour="blue", leave=False):
            mol = entry.molecule
            mol.normalise_labels()
            clone = mol.copy()
            clone.remove_atoms(a for a in clone.atoms if a.is_metal)
            # Work out which components to remove
            to_remove = [
                c
                for c in clone.components
                if self.is_solvent(c, solvents_smiles) and (remove_multidentate or not self.is_multidentate(c, mol))
            ]
            # Remove the atoms of selected components
            mol.remove_atoms(
                mol.atom(a.label) for c in to_remove for a in c.atoms
            )

            # If molecule have any atoms
            if mol.atoms:
                entry.crystal.molecule = mol
                list_entry.append(entry)

        if will_return:
            return list_entry
        else:
            self.list_target = list_entry

    @staticmethod
    def sorted_mol_atom(mol: Molecule, mode="global"):
        """
        Sorting Atoms in a Molecule
        :param mode: 'global' or 'local'
        :param mol:
        :return: List of atomic label.
        """
        sorter = GraphSort(mol)

        if mode == "global":
            return sorter.global_sorted()
        elif mode == "local":
            return sorted(mol.atoms, key=cmp_to_key(sorter.local_sorted), reverse=True)
        else:
            raise ValueError(f"The arg 'mode' not have value of {mode}, it only is allowed to be 'local' or 'global'")

    def split_crystals_to_metal_ligands_pairs_(self, *targets: Union[Entry, Crystal, Molecule],
                                               cm_symbol: Union[str, List[str]] = None,
                                               coordination_bond_type=1):
        """
        Split a crystal info to metal-ligand pairs (M-L pairs)
        :param targets: input Entry, Crystal or Molecule to split
        :param cm_symbol: the symbol of coordination metals (cm)
        :param coordination_bond_type: the bond type of generated coordination bond
        :return:
        """
        list_pair = []  # To store metal-ligand pairs
        targets = targets if targets else self.list_target  # Identify targets
        # Check the type of targets and transform them to molecules
        mols = self.transform_crystal_to_molecule(*targets)
        # Standardization of molecule
        mols = self.molecular_standardization(*mols, normalise_label=True, n_bar=False)
        # Transform cm_symbol to a list format and judge whether allowed all of metals as coordination metals
        cm_symbol, is_all_metals = self._cm_symbol(cm_symbol)

        # Splitting to metal-ligand pairs
        for mol in tqdm(mols, "Splitting crystal to metal-ligand pairs", colour="blue", leave=False):
            idt = mol.identifier  # the identifier (idt)
            pairs_count = 0  # Counts of metal-organic pairs in the mol
            # get labels of coordination metals (cm) and the set of their coordination atoms
            dict_cm_lbl = self._coordination_metal_with_their_coordination_atoms(mol, cm_symbol, is_all_metals)
            list_pair_mol = []
            for cpt in mol.components:  # for each component (cpt)
                clone = cpt.copy()  # a clone of the cpt
                clone.remove_atoms(a for a in clone.atoms if a.is_metal)  # remove all of metals in the clone
                ligands = clone.components
                for ligand in ligands:

                    # if this ligand contains only one atoms or not a organic molecule, pass
                    if len(ligand.atoms) < 2 or not ligand.is_organic:
                        continue

                    set_lal = {a.label for a in ligand.atoms}  # set of atomic labels of ligand (lal)
                    # for each label of coordination metal (lcm) and label coordination atom (set_lca)
                    for lcm, set_lca in dict_cm_lbl.items():
                        # Intersection set of label of coordination atoms (its_set_lca)
                        its_set_lca = set_lal & set_lca
                        # if its_set_lca is not empty, that the coordination bonds are exist
                        if its_set_lca:
                            pair = ligand.copy()  # Initialising a pair
                            cm = pair.add_atom(mol.atom(lcm))  # a coordination metal in the pairs
                            pairs_count += 1
                            # generation coordination bonds between cm and coordination atoms
                            for lca in its_set_lca:  # for each label of coordination atom in the intersection set
                                ca = pair.atom(lca)
                                pair.add_bond(coordination_bond_type, cm, ca)
                            # assign identifier for the metal-ligand pairs
                            pair.identifier = f"{cm.atomic_symbol}-{idt}-{pairs_count}"
                            list_pair_mol.append(pair)

            list_pair_mol = [SameMolContainer(p) for p in list_pair_mol]  # put pair into SameMolContainer
            list_pair_mol = self.merge_same_pairs(list_pair_mol, print_bar=False)
            list_pair.extend(list_pair_mol)

        self.list_target = self.merge_same_pairs(list_pair)

    @staticmethod
    def merge_same_pairs(ips: List, print_bar=True) -> List:
        """"""
        ops = []  # output SamMolContainer
        length = len(ips)

        if print_bar:
            process_bar = tqdm(total=length, desc="merge same mol", leave=False)

        dict_ips = {}
        for smc in ips:
            list_ips = dict_ips.setdefault(len(smc.mol.atoms), [])
            list_ips.append(smc)

        for an, ips in dict_ips.items():
            while ips:
                smc = ips.pop()  # pop a container
                to_pop_idx = []
                for idx, cp in enumerate(ips):  # comparative pair
                    if smc == cp:
                        to_pop_idx.append(idx)

                count = len(to_pop_idx)
                while to_pop_idx:
                    idx = to_pop_idx.pop()
                    smc.extend(ips.pop(idx))

                ops.append(smc)

                process_bar = locals().get("process_bar")
                if isinstance(process_bar, tqdm):
                    process_bar.update(count)

        return ops

    def put_mol_into_container(self):
        """    Put molecule target into SaveMolContainer class    """
        self.list_target = [SameMolContainer(m) for m in self.list_target]

    def write_bs_target(self, dir_pair_save: Path, dir_bs_save: Path):
        """"""
        if not dir_pair_save.exists():
            dir_pair_save.mkdir(parents=True)
        if not dir_bs_save.exists():
            dir_bs_save.mkdir(parents=True)
        for target in tqdm(self.list_target, "write_pair&bs", colour="#BBBB00", leave=False):
            assert isinstance(target, SameMolContainer)
            is_success = target.save_bs_sheet(dir_bs_save)
            # If save bond structure information is successful.
            if is_success:
                with io.MoleculeWriter(str(dir_pair_save.joinpath(f"{target.mol.identifier}.mol2"))) as writer:
                    writer.write(target.mol)

    @classmethod
    def statistics_bond_info(cls, dir_bs: Path, metal=False, num_ca=False, n_pair=False, diff_bl=False,
                             mean=False, median=False, dtb_median=False, std=False, blv=False):
        """
        Statistics about bond info
        :param dir_bs: dir of bond structure info save
        :param metal: Whether to statistic metal count
        :param num_ca: Whether to statistic number of coordination atoms for metal in a pairs
        :param n_pair: Whether to statistic number of pairs for each graph types
        :param diff_bl: Whether to statistic the diff between max and min bond length
        :param mean: Whether to statistic the bonds‘ mean values
        :param median: Whether to statistic the bonds' median
        :param dtb_median: Distribution (dtb) of difference between bonds and their median
        :param std: Whether to statistic the bonds' std
        :param blv: Whether to statistic the all of bond length for each bond types
        :return: list of statistic results
        """

        def statistics_metal_type():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "count metal", colour="blue", leave=False):
                stem = csv_file.stem
                sbl = stem.split("-")[0]
                count = dict_results.get(sbl, 0)
                dict_results[sbl] = count + 1

            return dict_results

        def statistics_num_ca():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics num ca", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                count = dict_results.get(nca, 0)
                dict_results[nca] = count + 1

            return dict_results

        def statistics_num_pairs():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics num pairs", colour="blue", leave=False):
                stem = csv_file.stem
                df = pd.read_csv(csv_file, index_col=0)
                dict_results[stem] = len(df)

            return dict_results

        def statistics_diff_bl():
            """"""
            list_results = []
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics_diff_bl", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for col in range(3, 1 + 3 * nca, 3):
                    column = df.iloc[:, col]
                    max_value = column.max()
                    min_value = column.min()
                    list_results.append(max_value - min_value)

            return list_results

        def statistics_bonds_mean():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics_bonds_mean", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for idx in range(nca):
                    mt = df.iloc[0, 1 + 3 * idx]
                    ca = df.iloc[0, 2 + 3 * idx]
                    bl = df.iloc[:, 3 + 3 * idx]
                    bl_mean = bl.mean()
                    list_mean = dict_results.setdefault(f"{mt}-{ca}", [])
                    list_mean.append(bl_mean)

            return dict_results

        def statistics_bonds_median():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics_bonds_median", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for idx in range(nca):
                    mt = df.iloc[0, 1 + 3 * idx]
                    ca = df.iloc[0, 2 + 3 * idx]
                    bl = df.iloc[:, 3 + 3 * idx]
                    bl_mean = bl.median()
                    list_median = dict_results.setdefault(f"{mt}-{ca}", [])
                    list_median.append(bl_mean)

            return dict_results

        def distribution_of_difference_between_bonds_and_their_median():
            """"""
            list_result = []
            for csv_file in tqdm(dir_bs.glob("*.csv"), "distribution_of_diff_bond_median", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for idx in range(nca):
                    bls = df.iloc[:, 3 + 3 * idx]
                    bl_median = bls.median()
                    for bl in bls:
                        list_result.append([abs(bl - bl_median), abs(bl - bl_median) / bl_median * 100])

            return list_result

        def statistics_bonds_std():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics_bonds_std", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for idx in range(nca):
                    mt = df.iloc[0, 1 + 3 * idx]
                    ca = df.iloc[0, 2 + 3 * idx]
                    bl = df.iloc[:, 3 + 3 * idx]
                    bl_mean = bl.std()
                    list_std = dict_results.setdefault(f"{mt}-{ca}", [])
                    list_std.append(bl_mean)

            return dict_results

        def statistics_bl_values():
            """"""
            dict_results = {}
            for csv_file in tqdm(dir_bs.glob("*.csv"), "statistics_bl_values", colour="blue", leave=False):
                df = pd.read_csv(csv_file, index_col=0)
                nca = df.iloc[0, 0]
                for idx in range(nca):
                    mt = df.iloc[0, 1 + 3 * idx]
                    ca = df.iloc[0, 2 + 3 * idx]
                    bl = df.iloc[:, 3 + 3 * idx]
                    for b in bl:
                        list_b = dict_results.setdefault(f"{mt}-{ca}", [])
                        list_b.append(b)

            return dict_results

        results = {}
        if metal:
            results["metal"] = statistics_metal_type()
        if num_ca:
            results["num_ca"] = statistics_num_ca()
        if n_pair:
            results["n_pair"] = statistics_num_pairs()
        if diff_bl:
            results["diff_bl"] = statistics_diff_bl()
        if mean:
            results["mean"] = statistics_bonds_mean()
        if median:
            results["median"] = statistics_bonds_median()
        if dtb_median:
            results["dtb_median"] = distribution_of_difference_between_bonds_and_their_median()
        if std:
            results["std"] = statistics_bonds_std()
        if blv:
            results["blv"] = statistics_bl_values()

        return results

    def statistic_ideal_bond_length(self, path_csv, *targets,
                                    required_elements: Union[str, Sequence[str]] = None,
                                    excluded_elements: Union[str, Sequence[str]] = None):
        """
        Statistic ideal bond length given by CSD
        :param path_csv: the path of csv to save results
        :param targets: output molecule targets
        :param required_elements: Element symbol or list of elements symbols
        :param excluded_elements: Element symbol or list of elements symbols
        :return:
        """

        targets = targets if targets else self.list_target

        if not required_elements:
            required_elements = []
        elif isinstance(required_elements, str):
            required_elements = [required_elements]

        if not excluded_elements:
            excluded_elements = []
        elif isinstance(excluded_elements, str):
            excluded_elements = [excluded_elements]

        required_metal = False
        excluded_metal = False
        required_nonmetal = False
        excluded_nonmetal = False

        if "metal" in required_elements:
            required_metal = True
        if "metal" in excluded_elements:
            excluded_metal = True
        if "nonmetal" in required_elements:
            required_nonmetal = True
        if "nonmetal" in excluded_elements:
            excluded_nonmetal = True

        dict_result = {}
        for target in tqdm(targets, "Statistic ideal bond length", colour="#00AAAA", leave=False):
            # Find target bonds
            mol = self.transform_crystal_to_molecule(target, n_bar=True)[0]
            if mol.all_atoms_have_sites:
                try:
                    mol.assign_bond_types()
                except:
                    continue
            else:
                continue
            if not required_elements:
                bonds = set(mol.bonds)
            elif required_metal:
                bonds = set([b for b in mol.bonds if any(a.is_metal for a in b.atoms)])
            elif required_nonmetal:
                bonds = set([b for b in mol.bonds if any(not a.is_metal for a in b.atoms)])
            else:
                bonds = set()

            bonds.update([b for b in mol.bonds if any(a.atomic_symbol in required_elements for a in b.atoms)])

            # Exclude bonds
            if excluded_metal:
                bonds = [b for b in bonds
                         if all((not a.is_metal) or (a.atomic_symbol in required_elements) for a in b.atoms)]
            elif excluded_nonmetal:
                bonds = [b for b in bonds
                         if all(a.is_metal or (a.atomic_symbol in required_elements) for a in b.atoms)]

            bonds = [b for b in bonds
                     if all((a.atomic_symbol not in excluded_elements)
                            or (a.atomic_symbol in required_elements) for a in b.atoms)]

            for bond in bonds:

                try:
                    a1, a2 = bond.atoms
                    bt = self.bt_format[bond.bond_type]
                    ibl = bond.ideal_bond_length
                    bn = f"{a1.atomic_symbol}{bt}{a2.atomic_symbol}" if a1.atomic_number > a2.atomic_number \
                        else f"{a2.atomic_symbol}{bt}{a1.atomic_symbol}"  # bond name
                    dict_result[bn] = ibl
                except:
                    continue

        # Save results
        ser = pd.Series(dict_result, name="ideal bond length")
        ser.sort_index(inplace=True)
        ser.to_csv(path_csv)

    def statistic_substructures(self, *targets, cm_symbol: Union[str, List[str]] = None):
        """
        Statistic substructures using predefined substructure Searchers
        :param targets: input targets. if not given, pull from self.list_target
        :param cm_symbol: the coordination metals symbol
        :return: Dict of statistic results
        """
        dict_sst, dict_c_sst = {}, {}
        targets = targets if targets else self.list_target
        # Transform cm_symbol to a list format and judge whether allowed all of metals as coordination metals
        cm_symbol, is_all_metals = self._cm_symbol(cm_symbol)
        # Check the type of targets and transform them to molecules
        mols = self.transform_crystal_to_molecule(*targets)
        for mol in tqdm(mols, "Statistic substructures", colour="blue", leave=False):
            idt = mol.identifier
            dict_sst[idt], dict_c_sst[idt] = self.statistic_sst_mol(mol, cm_symbol, is_all_metals)

        return dict_sst, dict_c_sst

    def statistic_sst_mol(self, mol, cm_symbol=None, is_all_metals=False):
        """
        statistics substructures (sst) for a molecule
        :param is_all_metals: Whether all metals are seen as coordination metals
        :param cm_symbol: coordination metal symbol
        :param mol: the molecule
        :return: dict of statistic results
        """

        # Statistics all substructures
        dict_mol_sst = {}  # dict of substructure
        set_hit_atoms = set()  # Recording labels of hit atoms
        for name, searcher in self.searchers.items():
            hits = []
            if not isinstance(searcher, list):
                hits, set_hit_atoms = self.statistics_sst_sms(hits, set_hit_atoms, mol, searcher)
            else:
                for ssc in searcher:  # for each sub-searcher (ssc)
                    hits, set_hit_atoms = self.statistics_sst_sms(hits, set_hit_atoms, mol, ssc)
            dict_mol_sst[name] = hits

        # Return all substructures only if cm_symbol and list_cm are not given
        # If only one of cm_symbol or list_cm is given, raise ValueError
        if not cm_symbol and not is_all_metals:
            return {n: len(h) for n, h in dict_mol_sst.items()}, None
        elif not (cm_symbol and is_all_metals):
            raise ValueError(
                "the args of 'cm_symbol' and 'list_cm' must be given simultaneously "
                "to statistics coordination substructures"
                "else, do not to be giving values for any of them"
            )

        # Configure information required for statistics coordination substructures
        # get labels of coordination metals (cm) and the set of their coordination atoms
        dict_cm_lbl = self._coordination_metal_with_their_coordination_atoms(mol, cm_symbol, is_all_metals)
        # get all of coordination atoms' label (all_ca_lbl) in the molecules
        set_all_ca = set(
            ca_lbl
            for set_ca_lbl in dict_cm_lbl.values()
            for ca_lbl in set_ca_lbl
        )

        # Statistics coordination substructures
        dict_mol_c_sst = {}  # dict of coordination substructure (c_sst)
        for name, hits in dict_mol_sst.items():
            c_hits = dict_mol_c_sst.setdefault(name, [])
            for hit in hits:
                # Get hit atoms label
                ha_lbl = set(a.label for a in hit.match_atoms())
                if ha_lbl & set_all_ca:
                    c_hits.append(c_hits)

        return dict_mol_sst, dict_mol_c_sst

    @staticmethod
    def statistics_sst_sms(hits, set_hit_atoms, mol, searcher):
        """
        statistics substructures (sst) for a molecule by single_searcher (sms)
        :param hits: list of hit substructures
        :param set_hit_atoms: set of atoms in hit substructures
        :param mol: the molecule to statistics
        :param searcher: the substructure searcher is used to statistics
        :return: hits and set_hit_atoms
        """
        for cpt in mol.components:
            cpt_hits = searcher.search(cpt)
            for cpt_hit in cpt_hits:
                if not set(a.label for a in cpt_hit.match_atoms()) & set_hit_atoms:
                    set_hit_atoms.update(a.label
                                         for a in cpt_hit.match_atoms()
                                         if a.atomic_symbol not in ["C", "H"]
                                         )
                    hits.append(cpt_hit)

        return hits, set_hit_atoms

    @staticmethod
    def transform_crystal_to_molecule(*targets: Union[Entry, Crystal, Molecule], n_bar=False):
        """
        Check the type of targets and transform them to molecules
        Only the Entry, Crystal and Molecule are supported in targets
        :return list of Molecule
        """
        mols = []
        bar = nothing_bar if n_bar else tqdm
        for target in bar(targets, "transform to molecule", colour="#00AAAA", leave=False):
            if isinstance(target, Molecule):
                mols.append(target)
            elif isinstance(target, (Entry, Crystal)):
                mols.append(target.molecule)
            else:
                raise TypeError(f"the {type(target)} class is not supported in target!")

        return mols

    def transform_mol_to_entry(self, *mols):
        """
        Transform Molecule objects into Entry
        :param mols: output list_mols
        :return: transformed entries if mols are given
        """
        will_return = True if mols else False
        mols = mols if mols else self.list_target
        entries = [
            Entry.from_molecule(m) for m in tqdm(mols, "transforming mol to entry", colour="#BBBB00", leave=False)
        ]

        if will_return:
            return entries
        self.list_target = entries

    def unique_identifier_mols(self, existing_idt: Set, *mols: Molecule):
        """"""
        # Check arguments
        if not isinstance(existing_idt, Set):
            raise TypeError("the arg existing_idt is not a Set!")

        will_return = True if mols else False
        mols = mols if mols else self.list_target

        dict_mols = {m.identifier: m for m in mols}
        set_idt = set(dict_mols.keys())
        set_idt.intersection_update(existing_idt)
        for idt in set_idt:
            _ = dict_mols.pop(idt)

        if set_idt:  # if some files are discarded to report
            print(f"\rdiscard {len(set_idt)} mols,Leave {len(dict_mols)} mols")
        existing_idt.update(dict_mols.keys())

        # Feedback results
        if will_return:
            return existing_idt, list(dict_mols.values())
        else:
            self.list_target = list(dict_mols.values())
            return existing_idt


class SameMolContainer(object):
    """    contain and compare Molecule   """
    # TODO: modify to allow not to perform filter and save all of pairs and bond infos.
    def __init__(self, mol: Molecule):
        """"""
        self.mol = mol
        self.same_mol = [mol]
        self.similarity_searcher = search.SimilaritySearch(mol)
        self.bs_sheet = None

    def __eq__(self, other):
        """"""
        if not isinstance(other, SameMolContainer):
            raise ValueError(f"{type(self)} can't compare to {type(other)}")

        if len(other.mol.atoms) == len(self.mol.atoms) and len(other.mol.bonds) == len(self.mol.bonds) and \
                (self.similarity_searcher.search_molecule(other.mol).similarity == 1.0):
            return True

        return False

    def extend(self, other):
        if not isinstance(other, SameMolContainer):
            raise ValueError(f"{type(other)} can't extend into {type(self)}")
        if any(not isinstance(m, Molecule) for m in other.same_mol):
            return ValueError(f"the {type(other)} contain non-molecule object(s)")
        self.same_mol.extend(other.same_mol)

    def normalize(self, assign_bond_type=False) -> bool:
        """
        Normalize all of molecule
        Args:
            assign_bond_type:

        Returns: <bool> Whether the normalization is performed successfully.
        """
        if assign_bond_type:
            for mol in self.same_mol:
                mol.assign_bond_types()
            self.mol.assign_bond_types()

        try:
            self.mol.add_hydrogens()
            self.mol.normalise_atom_order()
            self.mol.normalise_labels()
        except:
            return False

        for mol in self.same_mol:
            try:
                mol.add_hydrogens()
                mol.normalise_atom_order()
                mol.normalise_labels()
            except:
                return False

        return True

    def get_bs_sheet(self) -> bool:
        """
        Generate bond structure information sheet.
        The sheet <pd.DataFrame> save at self.bs_sheet
        Returns: <bool> Whether the molecular normalization are performed successfully,
                and get any correct bond informations.

        """
        # Normalise all of molecule save in the instance.
        normal_result = self.normalize()
        if not normal_result:
            return normal_result

        list_idt, list_bs = [], []
        na = len([m for m in self.mol.atoms if m.is_metal][0].neighbours)
        for pair in self.same_mol:
            m = [m for m in pair.atoms if m.is_metal][0]
            cas = m.neighbours
            p_na = len(cas)

            # Check pairs, to make sure they have save graphic structure.
            if na != p_na:
                continue

            bs = [p_na]
            get_null = False
            for a in cas:
                bond = pair.bond(m.label, a.label)
                ibl = bond.ideal_bond_length
                try:
                    bl = bond.length
                except:
                    get_null = True
                    continue
                bs.extend([ibl, a.label, bl])

            if get_null:
                continue
            list_idt.append(pair.identifier)
            list_bs.append(bs)

        # If None of correct bond information is got.
        if not list_idt or not list_bs:
            return False

        col = ["na"]
        for _ in range(na):
            col.extend(["ibl", "ca_label", "bl"])

        self.bs_sheet = pd.DataFrame(list_bs, index=list_idt, columns=col)

        return normal_result

    def save_bs_sheet(self, dir_save: Path) -> bool:
        """
        Save all of bond structure information into csv files.
        Args:
            dir_save:

        Returns:

        """
        is_success = self.get_bs_sheet()

        # If generate bond structure information is successful, then:
        if is_success:
            self.bs_sheet.to_csv(dir_save.joinpath(f"{self.mol.identifier}.csv"))
            return True
        return False


class GraphSort(object):
    """
    Giving a molecule, return a list of atoms satisfying the following condition:
        1) the first atom is the heaviest atom in the molecule;
        if there are multi heaviest atoms, the first atom is the atom whose heaviest neigh atom has biggest
        atomic number; if the heaviest atoms are same too, to compare the second, and so on;
        if there are multi heaviest atoms with exactly same neigh atoms, to compare the atoms in heaviest atoms
        and so on.
        2) sorting atoms by step up one by one from the first atoms (0-step) to it's neigh atoms (1-step) to
        the final atoms (n-step).
        for atoms with same steps, Prioritizing them by 1)
    """

    def __init__(self, mol: Molecule, normalize_mol=False, sort_bond=False):
        """"""
        self.mol = mol
        # The order of bonds, where Unknown, Delocalised and Pi bonds do not have physical meaning!
        self.bond_type = dict(
            Unknown=0,
            Single=1,
            Aromatic=1.5,
            Double=2,
            Triple=3,
            Quadruple=4,
            Delocalised=7,
            Pi=9,
        )
        self.sort_bond = sort_bond
        if normalize_mol:
            self.normalize_mol()
        self.al = [a.label for a in mol.atoms]
        # The result of comparison, default values are 0 for pairs of atoms if have not to compared
        self.df_sorted = pd.DataFrame(np.zeros((len(self.al), len(self.al))), index=self.al, columns=self.al)
        # The pair of atoms had compared
        self.compared = pd.DataFrame(np.zeros((len(self.al), len(self.al))), index=self.al, columns=self.al)

    def global_sorted(self):
        """"""
        fa = sorted([a for a in self.mol.atoms], key=cmp_to_key(self.local_sorted), reverse=True)[0]
        sorted_atoms = [fa.label]
        current_atoms = sorted(
            [a for a in fa.neighbours if a.label not in sorted_atoms],
            key=cmp_to_key(self.local_sorted),
            reverse=True
        )
        length = len(self.al)
        while len(sorted_atoms) < length:
            sorted_atoms.extend([a.label for a in current_atoms])
            current_atoms = [
                a
                for ca in current_atoms
                for a in sorted(
                    [a for a in ca.neighbours if a.label not in sorted_atoms],
                    key=cmp_to_key(self.local_sorted),
                    reverse=True
                )
            ]
        return sorted_atoms

    def local_sorted(self, a1: Atom, a2: Atom) -> int:
        """
        Comparing the local priorities of two atoms in a molecule,
        priorities are determined as following steps:
        1) Atoms with large atomic numbers have higher priority;
        2) If the atomic numbers of the two atoms (called the current comparison atom, CCAs) are the same,
            the priorities of the adjacent atoms of the two atoms are sorted from high to bottom using this method,
            and the atomic numbers of the atoms in the same position in the two atomic sequences (called the
            second comparison atom, SCAs) and the bond levels with the CCAs are compared.
            If the atomic number and bond level of a second atom comparison are found to be different,
            the priority between CCAs is equal to the comparison result of the atomic number and
            bond level between the SCAs. If the comparing results of all the second atoms in the atomic
            number and bond level is the same, the priority of the atoms in the same position in the two atomic
            sequences is compared, and the comparison result is the priority of the CCAs.
        3) When the CCAs are being compared,  the comparison request has been proposed before,
            it shows that the two atoms are completely equal.
        """
        # a1 and a2 have compared.
        if self.compared.loc[a1.label, a2.label]:
            return self.df_sorted.loc[a1.label, a2.label]

        self.compared.loc[a1.label, a2.label] = self.compared.loc[a2.label, a1.label] = 1
        if self.sorted_atomic_number(a1, a2):
            self.df_sorted.loc[a1.label, a2.label] = self.sorted_atomic_number(a1, a2)
            self.df_sorted.loc[a2.label, a1.label] = self.sorted_atomic_number(a2, a1)
            return self.sorted_atomic_number(a1, a2)
        else:
            nei_a1 = sorted(a1.neighbours, key=cmp_to_key(self.local_sorted), reverse=True)
            nei_a2 = sorted(a2.neighbours, key=cmp_to_key(self.local_sorted), reverse=True)

            for na1, na2 in zip(nei_a1, nei_a2):
                if self.sorted_atomic_number(na1, na2):
                    return self.sorted_atomic_number(na1, na2)

                # Comparing bond order
                elif self.sort_bond:
                    b1 = self.mol.bond(a1.label, na1.label)
                    b2 = self.mol.bond(a2.label, na2.label)

                    if self.sorted_bond_type(b1, b2):
                        self.df_sorted.loc[a1.label, a2.label] = self.sorted_bond_type(b1, b2)
                        self.df_sorted.loc[a2.label, a1.label] = self.sorted_bond_type(b2, b1)
                        return self.sorted_bond_type(b1, b2)

            if len(nei_a1) > len(nei_a2):
                self.df_sorted.loc[a1.label, a2.label] = 1
                self.df_sorted.loc[a2.label, a1.label] = -1
                return 1
            elif len(nei_a1) < len(nei_a2):
                self.df_sorted.loc[a1.label, a2.label] = -1
                self.df_sorted.loc[a2.label, a1.label] = 1
                return -1

            else:
                for na1, na2 in zip(nei_a1, nei_a2):
                    return self.local_sorted(na1, na2)

    @staticmethod
    def sorted_atomic_number(a_1, a_2):
        """"""
        if a_1.atomic_number > a_2.atomic_number:
            return 1
        elif a_1.atomic_number < a_2.atomic_number:
            return -1
        else:
            return 0

    def sorted_bond_type(self, b1, b2):
        """"""
        b1_order = self.bond_type[b1.bond_type]
        b2_order = self.bond_type[b2.bond_type]

        if b1_order == b2_order:
            return 0
        elif b1_order > b2_order:
            return 1
        else:
            return -1

    def normalize_mol(self):
        """    Normalizing molecule before sorting    """
        try:
            self.mol.assign_bond_types()
            self.mol.normalise_atom_order()
            self.mol.add_hydrogens()
        except:
            print(self.mol.identifier)


class SortedAtom(Atom):
    """"""

    def __init__(self, atomic_symbol='', atomic_number=0, coordinates=None, label='', formal_charge=0, _atom=None):
        super(SortedAtom, self).__init__(
            atomic_symbol, atomic_number, coordinates, label, formal_charge, _atom
        )
        self.mol = Molecule(_molecule=self._molecule)
        self.al = [a.label for a in self.mol.atoms]
        # The result of comparison, default values are 0 for pairs of atoms if have not to compared
        self.df_sorted = pd.DataFrame(np.zeros((len(self.al), len(self.al))), index=self.al, columns=self.al)
        # The pair of atoms had compared
        self.compared = pd.DataFrame(np.zeros((len(self.al), len(self.al))), index=self.al, columns=self.al)

    def __eq__(self, other):
        """="""

    def sorted(self, other: Atom):
        """"""

    @classmethod
    def from_atom(cls, a: Atom):
        """"""
        sa = cls(_atom=a._atom)
        return sa

    @staticmethod
    def sorted_atomic_number(a_1, a_2):
        """"""
        if a_1.atomic_number > a_2.atomic_number:
            return 1
        elif a_1.atomic_number < a_2.atomic_number:
            return -1
        else:
            return 0
