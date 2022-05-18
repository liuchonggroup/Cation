"""
python v3.7.9
@Project: unique_pairs_and_bond_structure.py
@File   : 1_mine_crystals.py
@Author : Zhiyuan Zhang
@Date   : 2021/12/21
@Time   : 9:24
"""
from pathlib import Path
from module.csd_miner import CSDMiner

dir_root = Path.cwd().parent
path_id = dir_root.joinpath("input_data/identifier.xlsx")
metals = ["Sr", "Cs"]

CSDMiner.get_identifier_for_each_metals(path_id, *metals)
