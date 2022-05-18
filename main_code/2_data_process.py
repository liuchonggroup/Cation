"""
python v3.7.9
@Project: cation_new
@File   : 2_data_process.py
@Author : Zhiyuan Zhang
@Date   : 2022/5/11
@Time   : 15:44
"""
from pathlib import Path
from module.graph_data import SelectYDescriptor, BondNumberFilters, IMGraphDataset
from module.csd_process import MetalLigandPairProcessor

if __name__ == "__main__":

    dir_root = Path.cwd().parent
    # Arg Setting
    struct_save_dir = "sr_cs"
    ms = ["Cs", "Sr"]  # Metals
    split_ML = False
    syd = SelectYDescriptor(2)
    bnf = BondNumberFilters(1, 1)

    # input data directory
    dir_input_data = dir_root.joinpath("input_data")
    path_ar = dir_input_data.joinpath("atomic_radius.xlsx")
    path_id = dir_input_data.joinpath("identifier.xlsx")
    dir_ag = dir_input_data.joinpath("anion")  # dir anion gas
    dir_con = dir_input_data.joinpath("functional_group")

    # dir data
    dir_d = dir_root.joinpath(f"results/{struct_save_dir}")

    # Args for generating virtual Metal-Ligand pairs
    selected_fg = [
        'B1_Aliphatic_alcohol', 'E3_Sulfone', 'C2_carbonyl', 'C1_ether',
        'E4_sulfonic_acid', 'B1_mercaptan', 'F1_pyridine', 'D3_boronic_acid',
        'D1_Tertiary_amine', 'E4_phosphate', 'E1_Imine', 'D1_ketone',
        'G1_phenol', 'B1_Cyano', 'C2_nitroso'
    ]
    metals = ["Sr", "Cs"]
    sorted_ca = [["O"], ["N"], ["P"], ["S"]]

    # Initialize data processor
    processor = MetalLigandPairProcessor(
        metals=metals,
        coordinate_functional_groups=selected_fg,
        sorted_coordinate_atoms=sorted_ca,
        dir_data=dir_d,
        dir_con_files=dir_con,
        path_identifier=path_id,
        dir_anion_gas=dir_ag,
        path_atomic_radii=path_ar,
        split_metal_ligand=split_ML,
        dataset_type=IMGraphDataset,
    )

    # Processing
    processor.process(pre_filter=bnf, transform=syd)
