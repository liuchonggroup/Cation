"""
python v3.7.9
@Project: unique_pairs_and_bond_structure.py
@File   : 3_training_pred.py
@Author : Zhiyuan Zhang
@Date   : 2021/12/21
@Time   : 10:50
"""
from pathlib import Path
from torch.nn.functional import l1_loss, mse_loss, huber_loss
from torch.optim import Adam, AdamW, SGD
from module.graph_data import SelectYDescriptor, BondNumberFilters, IMGraphDataset
from module.training import TrainModelT
from module.models import Model

# Set hyperparameters
learning_rate = 0.001  # learning rate, space: [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
loss_fn = mse_loss  # loss function, space: [l1_loss, mse_loss, huber_loss]
optimizer = Adam  # optimizer, space: [Adam, AdamW, SGD]
gcn_layers = 6  # GCN layers, space: [3, 4, 5, 6 ,7 ,8]
fcn_layers = 4  # FCN layers, space: [2, 3, 4, 5]
# pooling layers location, space::
#   [(1, 2)] if GCN Layers = 3,
#   [(1, 3), (2, 3)] if GCN Layers = 4,
#   [(2, 4), (3, 4)] if GCN Layers = 5,
#   [(2, 5), (3, 5)] if GCN Layers = 6,
#   [(3, 5), (3, 6), (4, 6)] if GCN Layers = 7,
#   [(3, 5), (3, 6), (4, 6), (4, 7) (5, 7)] if GCN Layers = 8
pooling_local = (3, 5)
pooling_ratio1 = 0.65  # Pooling ratio1, space: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
pooling_ratio2 = 0.5  # Pooling ratio2, space: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
max_epochs = 3000
device = 0  # which cuda to be used to training; for cpu, it's None
k = None  # k-Fold cross validation


if __name__ == "__main__":
    dir_root = Path.cwd().parent
    dataset_name = "sr_cs"
    pred_data = ""
    model_name = f"(n{dataset_name})(g{gcn_layers}f{fcn_layers})(e{max_epochs})(lr{learning_rate})(d{device})(k{k})(lh)"
    syd = SelectYDescriptor(2)
    bnf = BondNumberFilters(1, 1)
    dir_model = dir_root.joinpath(f"results/{dataset_name}/model/{model_name}")
    dir_dataset = dir_root.joinpath(f"results/{dataset_name}/dataset")
    dataset = IMGraphDataset(dir_dataset, pre_filter=bnf, transform=syd)
    tm = TrainModelT(dir_model, dataset, model_name=model_name, k_fold=k)
    tm.model = Model(
        gcn_layers=gcn_layers,
        fcn_layers=fcn_layers,
        dataset=dataset,
        pool_local=pooling_local,
        pool_ratio1=pooling_ratio1,
        pool_ratio2=pooling_ratio2,
    )
    tm.auto_split = True
    tm.loss_fn = loss_fn
    tm.device_idx = device
    tm.test_size = 0.1
    tm.max_epoch = max_epochs
    tm.batch_size = 256
    tm.set_opti_args(optimizer, lr=learning_rate, weight_decay=4e-5)
    tm.save_id = True
    tm.save_best_model = True
    tm.early_stop_count = 250
    tm.training()

    tm.dataset = IMGraphDataset(dir_root.joinpath(f"results/{dataset_name}/v_dataset"), pre_filter=bnf, transform=syd)
    tm.load_trained_model()
    pred, ideal_bond_length, idt = tm.predict(all_data=True)
    relative_bond_length = pred / ideal_bond_length
