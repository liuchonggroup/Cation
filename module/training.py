"""
python v3.7.9
@Project: 1-2.split_crystals_to_pairs.py
@File   : training_beta.py
@Author : Zhiyuan Zhang
@Date   : 2022/3/11
@Time   : 16:34
"""

import math
import time
import copy
from pathlib import Path
from typing import Dict, Callable, List, Union, TypeVar, Iterable, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fn
from sklearn.model_selection import KFold, train_test_split
from torch import Tensor
from torch.nn import Module, Linear, Sequential
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

T = TypeVar("T")


class Model(Module):
    """"""
    def __init__(
            self, model: Module, append_layers: Dict[str, Module],
            replace_layers: Dict[str, Module], insert_layers: Dict[str, Tuple[Union[str, Module]]]
    ):
        super(Model, self).__init__()
        for module_name, module in model.named_children():

            if module_name in insert_layers:
                name, layer = insert_layers.get(module_name)
                setattr(self, name, layer)

            if module_name in replace_layers:
                layer = replace_layers.get(module_name)
                setattr(self, module_name, layer)
            else:
                setattr(self, module_name, module)

        for append_name, layer in append_layers.items():
            setattr(self, append_name, layer)

    def forward(self, x) -> Tensor:
        """"""
        for name, layer in self.named_children():
            x = layer(x)

        return x


def l2_sample_error(pred: Tensor, target: Tensor) -> Tensor:
    """"""
    return (pred - target) ** 2


# TODO: the following code have some error.
class GraphAdaBoost(object):
    """"""
    def __init__(self, train_set, batch_size: int = 1024, sample_loss=l2_sample_error):
        """"""
        self.train_set = train_set
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.idt = self.__init_idt()
        # Logarithmic weight for samples
        self.ws = torch.ones(len(train_set), requires_grad=False).to(self.device)
        self.bms = []  # list of base models
        self.wm = []  # list of weight for base models
        self.sample_loss = sample_loss
        self.max_error = 0

    def fit(self, base_model: Module):
        """"""
        # Predication from base predictor
        idt, pred, target = self.__base_predict(base_model)
        ei = (pred - target) ** 2
        max_error = torch.max(ei)
        ei = ei / max_error  # normalize absolute error into [0, 1]
        # The total error ratio
        e = torch.dot(self.ws / self.ws.sum(), ei)
        beta = e / (1 - e)  # a measure of confidence in the predictor
        self.ws = self.ws + torch.log(torch.pow(beta, 1 - ei))

        self.idt = idt

        self.bms.append(base_model)
        self.wm.append(beta)

    def loss_fn(self):
        """"""
        loss_fn = SampleWeightedLoss(
            dict((i, w) for i, w in zip(self.idt, self.ws)),
            sample_loss=self.sample_loss  # l2_sample_loss as default
        )
        return loss_fn

    def __base_predict(self, base_model: Module):
        """"""
        list_idt, list_pred, list_target = [], [], []
        loader = iter(DataLoader(self.train_set, self.batch_size))
        with torch.no_grad():
            for batch in loader:
                batch.cuda()
                list_idt.extend(batch.idt)
                list_pred.append(base_model(batch).flatten())
                list_target.append(batch.y)

        return list_idt, torch.cat(list_pred), torch.cat(list_target)

    def __init_idt(self) -> List:
        """    Return the list of sample idt    """
        list_idt = []
        loader = iter(DataLoader(self.train_set, self.batch_size))
        for batch in loader:
            list_idt.extend(batch.idt)

        return list_idt

    def predict(self, batch):
        """"""
        weight = sum([torch.log(1 / w) for w in self.wm])
        median, indices = torch.median(torch.stack(self.wm) * torch.cat([m(batch) for m in self.bms], dim=1), dim=1)
        return weight * median


class SampleWeightedLoss(_Loss):
    """"""
    def __init__(self, sample_weight: Dict, sample_loss: Callable, device="cuda"):
        """"""
        super(SampleWeightedLoss, self).__init__()
        self.log_ws = sample_weight  # Logarithmic sample weight. If the actual weight is 0.1, giving a -1.
        self.loss = sample_loss
        self.log_num_sample = math.log(len(sample_weight))
        self.device = device

    def forward(self, pred: Tensor, target: Tensor, samples_id: List) -> Tensor:
        """"""
        log_batch_size = math.log(len(pred))
        samples_loss = self.loss(pred, target)
        log_samples_weight = \
            torch.tensor([self.log_ws[i] for i in samples_id]) + self.log_num_sample - log_batch_size
        log_samples_weight = log_samples_weight.cuda() if self.device == "cuda" else samples_loss.cpu()

        return torch.dot(torch.exp(log_samples_weight), samples_loss)


class MetalSampler(object):
    """"""
    def __init__(self, data_source: Union[Dataset, List[Data]], random_=True, exclude_elements: List = None):
        """"""
        if not exclude_elements:
            self.dataset = data_source
        else:
            self.dataset = [d for d in data_source if d.idt.split("-")[0] not in exclude_elements]
        self.random = random_
        self.ele_idx = self.get_element()

    def get_element(self):
        """"""
        data_ele = [d.idt.split("-")[0] for d in self.dataset]
        ele_idx = {}
        for i, e in enumerate(data_ele):
            idx = ele_idx.setdefault(e, [])
            idx.append(i)

        return ele_idx

    def __iter__(self):
        """"""
        if self.random:
            return self.random_sample()

    def random_sample(self):
        """"""
        indices = []
        class_count = len(self.dataset) // len(self.ele_idx)
        residue = len(self.dataset) - class_count * len(self.ele_idx)

        for e, list_idx in self.ele_idx.items():
            random_sample_from_list_idx = torch.randint(0, len(list_idx), [class_count])
            indices.append(torch.tensor(list_idx)[random_sample_from_list_idx])

        if residue:
            indices.append(torch.randint(0, len(self.dataset), [residue]))

        indices = torch.cat(indices, dim=0)
        rp = torch.randperm(indices.shape[0])

        return iter(indices[rp])


class DataSlitter(object):
    """"""
    def __init__(self, dataset: Union[List, Dataset], k_fold=None, test_size=0.1, classifier: Callable = None):
        """"""
        self.dataset = dataset
        self.train_dataset = None
        self.test_dataset = None
        self.k = k_fold
        self.k_fold = self._k_fold(k_fold)
        self.__kFold_data = None
        self.test_size = test_size
        self.classifier = classifier

    def k_fold_split(self):
        """"""
        if self.classifier is None:
            self.__kFold_data = self.non_classed()
        else:
            self.__kFold_data = self.classed()

    @property
    def k_fold_data(self):
        """    Return the generate to generate train data and test data    """
        return self.__kFold_data

    def classed(self):
        """"""
        dict_k = {}
        for classed_data in self.get_dataset:
            for k, (train_idx, test_idx) in enumerate(self.k_fold.split(classed_data)):
                train_, test_ = dict_k.setdefault(k, ([], []))
                for idx in train_idx:
                    train_.append(classed_data[idx])
                for idx in test_idx:
                    test_.append(classed_data[idx])

        return (
            (k, train_data, test_data) for k, (train_data, test_data) in dict_k.items()
        )

    def non_classed(self):
        """"""
        for k, (train_idx, test_idx) in enumerate(self.k_fold.split(self.dataset)):
            train_data, test_data = self.dataset[list(train_idx)], self.dataset[list(test_idx)]
            yield k, train_data, test_data

    @staticmethod
    def _k_fold(k_fold: int) -> Union[None, KFold]:
        """"""
        if k_fold is None:
            return None
        elif isinstance(k_fold, int) and k_fold > 1:
            return KFold(k_fold, shuffle=True)
        else:
            raise ValueError("the arg k_fold is required to be integer more than 1")

    def split_dataset(self):
        """
        split dataset into test set and train set, if self.k_fold is not None, perform k_
        :return: 2 lists: test set, train set
        """
        if self.k_fold:
            k, train_data, test_data = next(self.__kFold_data)
            print(f"{k}_fold dataset split!")
            self.train_dataset, self.test_dataset = train_data, test_data

        else:
            if self.classifier is None:
                self.test_dataset, self.train_dataset = train_test_split(self.dataset, train_size=self.test_size)
            else:
                test_dataset, train_dataset = [], []
                for classed_data in self.get_dataset:
                    test_data, train_data = train_test_split(classed_data, train_size=self.test_size)
                    test_dataset.extend(test_data)
                    train_dataset.extend(train_data)

                self.test_dataset, self.train_dataset = test_dataset, train_dataset

    @property
    def get_dataset(self):
        """"""
        return self.classifier(self.dataset) if self.classifier else self.dataset

    def __getitem__(self, item: Union[int, List[int], Iterable]):
        """"""
        return self.dataset[item]

    def __repr__(self):
        """"""
        return f"DataSlitter:\n" \
               f"dataset: {self.dataset}\n" \
               f"train dataset: {self.test_dataset}\n" \
               f"test dataset: {self.test_dataset}\n" \
               f"k_fold: {self.k_fold}\n" \
               f"classifier: {self.classifier}"


def metal_classifier(dataset: Iterable):
    """"""
    dict_class = {}
    for data in dataset:
        metal = data.idt.split("-")[0]
        list_data = dict_class.setdefault(metal, [])
        list_data.append(data)

    return list(dict_class.values())


class ModelContainer(object):
    """    This class is used to generate the desired module during automatic training    """

    def __init__(self, model: Module):
        """"""
        if not isinstance(model, Module):
            TypeError("the arg model need to be a Module!")
        self.model = model
        self.train_model = None
        self.model_module_names = [n for n, m in model.named_modules()]
        self.children = {}
        self.append_layers = {}
        self.replace_layers = {}
        self.insert_layers = {}
        self._grad_layers = None
        self._exclude_grad_layers = []
        self.import_path = None
        self.save_best_model = False
        self.best_test_loss = None
        self.path_model_save = None
        self.path_dict_save = None

    def __repr__(self) -> str:
        """"""
        return f"ModelInitializer:\n{self.M}"

    def append_layer(self, layers: Union[List[Module], Module], names: Union[List[str], str] = None):
        """"""
        def merge_name_layer() -> None:
            """"""
            if not isinstance(layer, Module):
                raise TypeError(f"all of input layers are required to be Module, but get a {type(layer)}")
            if names is None:
                self.append_layers[f"add_layer{len(self.append_layers)}"] = layer
            elif isinstance(name, str):
                if name not in self.model_module_names:
                    self.append_layers[name] = layer
                else:
                    raise NameError("the append layer can't name to be which have been existed in original model!")
            else:
                raise TypeError(f"The names is required to be str or None, instead of {name}")

        if names is None or isinstance(names, str):
            if not isinstance(layers, Module):
                raise ValueError("layers must be a Module when name is None or string")
            layer, name = layers, names
            merge_name_layer()

        elif isinstance(names, list) and isinstance(layers, list) and len(names) == len(layers):
            for name, layer in zip(names, layers):
                merge_name_layer()

        else:
            raise ValueError("both names and layers should be list with same length when names is not None or string!")

    def config_grad(self):
        """    Configure specific layers require gradient calculation    """
        assert isinstance(self.M, Module)
        if self._grad_layers is None:  # No action
            pass
        elif self._grad_layers is True:  # All layers can be trained
            self.M.requires_grad_(True)
        elif self._grad_layers is False:  # All layers can't be trained
            self.M.requires_grad_(False)
        else:
            # Specific layer to be choose can be trained
            # Check Layer names
            module_names = [n for n, m in self.M.named_modules()]
            para_names = [n for n, p in self.M.named_parameters()]
            for layer_name in self._grad_layers:
                if (layer_name not in module_names) and (layer_name not in para_names):
                    raise ValueError(
                        f"layer or para {layer_name} not in the model"
                        f"the model have layers named:\n {','.join([n for n in module_names])}\n"
                        f"the model have paras named:\n {','.join([n for n in para_names])}"
                    )

            self.M.requires_grad_(False)

            # Allow some layers to grad
            for layer_name in self._grad_layers:
                layer = self.M.get_submodule(layer_name)
                layer.requires_grad_(True)

            for name, para in self.M.named_parameters():
                if name in self._grad_layers:
                    para.requires_grad_(True)

            # Exclude submodule to grad
            for layer_name in self._exclude_grad_layers:
                layer = self.M.get_submodule(layer_name)
                layer.requires_grad_(False)

            for name, para in self.M.named_parameters():
                if name in self._exclude_grad_layers:
                    para.requires_grad_(False)

    def insert_layer(self, layer: Module, name: str, bottom_layer: str) -> None:
        """

        Args:
            layer: inserted layer
            name: The name of inserting layer
            bottom_layer: The layer following the inserted layer, which is exist in original module

        Returns:

        """
        if isinstance(layer, Module) and isinstance(name, str) and isinstance(bottom_layer, str):
            assert bottom_layer in self.model_module_names
            assert name not in self.model_module_names
            self.insert_layers[bottom_layer] = (name, layer)

    @classmethod
    def load_model(cls, path_model: Path) -> "ModelContainer":
        """"""
        model = torch.load(path_model)
        output = cls(model)
        output.import_path = path_model
        return output

    def load_current_model(self):
        """"""
        if self.save_best_model:
            self.M.load_state_dict(torch.load(self.path_dict_save))

    def named_children(self) -> Dict[str, Module]:
        """"""
        for name, layer in self.model.named_children():
            self.children[name] = ModelContainer(layer)

        return self.children

    @property
    def M(self) -> Module:
        """"""
        if not self.train_model:
            self.reassemble_model()

        return self.train_model

    def replace_layer(self, layers: Union[List[Module], Module], names: Union[List[str], str]):
        """"""
        if isinstance(layers, Module) and isinstance(names, str):
            assert names in self.model_module_names
            self.replace_layers[names] = layers

        elif isinstance(layers, list) and isinstance(names, list) and len(layers) == len(names):
            for name, layer in zip(names, layers):
                assert isinstance(name, str) and isinstance(layers, Module)
                assert name in self.model_module_names
                self.replace_layers[name] = layer

        else:
            raise ValueError("both names and layers should be list with same length when names is not None or string!")

    def reassemble_model(self) -> None:
        """"""
        if self.append_layers and self.replace_layers and self.insert_layers:
            self.train_model = copy.deepcopy(
                Model(self.model, self.append_layers, self.replace_layers, self.insert_layers)
            )
        else:
            self.train_model = copy.deepcopy(self.model)

    def set_grad_layers(self, grad_layers: Union[None, List[str], bool], exclude_grad_layers: List[str] = None):
        """    Assign layers are allowed to calculate gradient when training    """
        self._grad_layers = grad_layers
        self._exclude_grad_layers = exclude_grad_layers if exclude_grad_layers else []

    def update_model(self) -> None:
        """"""
        self.reassemble_model()

    def save_model(self, save_dir: Path, k: int, test_loss: Union[int, Tensor, list] = None) -> None:
        """
        Save model and model state dict onto dist
        Args:
            save_dir: directory to save Save model and model state dict
            k: k fold counts
            test_loss: test loss

        Returns: None

        """
        if (test_loss is None) or (self.best_test_loss is None) or self.best_test_loss > test_loss:

            # Save total model and state_dict
            if not self.import_path or self.import_path.parent != save_dir:
                torch.save(self.M, save_dir.joinpath(f"model_k{k}.pth" if k is not None else "model.pth"))
                torch.save(
                    self.M.state_dict(),
                    save_dir.joinpath(f"state_dict_k{k}.pth" if k is not None else "state_dict.pth")
                )
                self.path_model_save = save_dir.joinpath(f"model_k{k}.pth" if k is not None else "model.pth")
                self.path_dict_save = save_dir.joinpath(f"state_dict_k{k}.pth" if k is not None else "state_dict.pth")
            else:
                raise RuntimeError(f"DANGER: Overwriting the trained model!!!"
                                   f"path: {str(self.import_path)}")

            # If the self.best_test_loss is not None,
            # the model will not be stored every time unless the loss function for the test set becomes lower.
            if self.save_best_model:
                self.best_test_loss = test_loss


class OptimizerTool(object):
    """"""

    def __init__(
            self, opti_type: T = torch.optim.Adam, lr: float = 1e-3,
            sch_type: T = None, model: Model = None, **opti_kwargs
    ):
        self.opti_type = opti_type
        self.lr = lr
        self.sch_type = sch_type
        self.opti_kwargs = opti_kwargs
        self.sch_kwargs = None
        self.optimizer = None
        self.scheduler = None
        self.model = model

    def config_optimizer(self, model: Module):
        """"""
        self.optimizer = self.opti_type(model.parameters(), lr=self.lr, **self.opti_kwargs)

    def config_scheduler(self):
        """"""
        if self.sch_type and self.sch_type:
            self.scheduler = self.sch_type(self.optimizer, **self.sch_kwargs)
            print("lr_scheduler configured!")

    def backpropagation(self, loss: _Loss):
        """    Backpropagation    """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def scheduler_step(self):
        """"""
        if self.scheduler:
            self.scheduler.step()

    def set_opti_args(self, opti_type: T = None, lr: float = None, **kwargs):
        """"""
        if opti_type:
            self.opti_type = opti_type
        if lr:
            self.lr = lr
        if kwargs:
            self.opti_kwargs = kwargs

    def set_sch_args(self, sch_type: T = None, **kwargs):
        """"""
        if sch_type:
            self.sch_type = sch_type
        if kwargs:
            self.sch_kwargs = kwargs


class ArgSetter(object):
    """    Setting TrainModel parameters by this class    """
    def __init__(
            self,
            config_kwargs: Dict[int, Dict[str, Union[None, List[str], bool]]]
    ):
        """"""
        self.kwargs = config_kwargs

    def __call__(self, tm, epoch: int):
        """"""
        kwargs = self.kwargs.get(epoch, None)
        if kwargs:
            tm.set_grad_layers(**kwargs)
            tm.config_grad()
            tm.print_para_grad()


class TrainModel(object):
    """"""
    def __init__(self, root: Path, dataset=None, k_fold: int = None, model_name="model"):
        """"""
        self.root = root
        self.model_name = model_name
        self.dataset = dataset
        self.max_epoch = 1000
        self.batch_size = 128
        self.loss_fn = torch.nn.MSELoss()
        self.device_idx = None
        self.k_fold = self._k_fold(k_fold)
        self.test_size = None
        self.test_dataset = None
        self.train_dataset = None
        self.mContainer = None
        self.optimizer_tool = OptimizerTool()
        self.__kFold_data = None
        self.save_id = False
        self.auto_split = False  # weather auto split dataset to train_set and text_set
        self.loss_curve = []
        self.is_transfer = False
        self.args_setter = None
        self.data_sampler: Union[None, type] = None

    def __repr__(self):
        """"""
        return f"----------------Train Neural Network Tool----------------\n" \
               f"\tModelName: {self.model_name}\n" \
               f"\tModelRoot: {self.root}\n" \
               f"\tDataset: {self.dataset}\n" \
               f"\tMaxEpoch: {self.max_epoch}\n" \
               f"\tBatchSize: {self.batch_size}\n" \
               f"\tDevice: {'cpu' if self.device_idx is None else 'cuda:' + str(self.device_idx)}\n" \
               f"\tTestSize: {self.test_size}\n" \
               f"\tTransferTrain: {self.is_transfer}\n" \
               f"\tSaveID: {self.save_id}\n" \
               f"\tModel: {self.M}\n"

    @property
    def model(self) -> Union[Module, None]:
        """"""
        if self.mContainer is None:
            return None
        else:
            assert isinstance(self.mContainer, ModelContainer)
            return self.M

    @model.setter
    def model(self, model: Model):
        """"""
        self.mContainer = ModelContainer(model)

    def append_layers(self, layers: Union[List[Module], Module], names: Union[List[str], str]):
        """    Append new module layer after current model    """
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.append_layer(layers, names)

    def config_grad(self):
        """    Determine which layers require gradient calculation    """
        self.mContainer.config_grad()

    def init_model(self):
        """"""
        self.mContainer.reassemble_model()
        self.init_model_arg()

    def set_opti_args(self, opti_type: T = None, lr: float = None, **kwargs):
        """"""
        self.optimizer_tool.set_opti_args(opti_type, lr, **kwargs)

    def set_sch_args(self, sch_type: T, **kwargs):
        """"""
        self.optimizer_tool.set_sch_args(sch_type, **kwargs)

    @staticmethod
    def _k_fold(k_fold: int) -> Union[None, KFold]:
        """"""
        if k_fold is None:
            return None
        elif isinstance(k_fold, int) and k_fold > 1:
            return KFold(k_fold, shuffle=True)
        else:
            raise ValueError("the arg k_fold is required to be integer more than 1")

    def init_model_arg(self):
        """    initialize model's arguments    """
        if self.is_transfer:
            return

        for submodule in self.M.modules():
            if hasattr(submodule, "weight") and submodule.weight is not None:
                torch.nn.init.orthogonal_(submodule.weight)
            if hasattr(submodule, "bias") and submodule.bias is not None:
                torch.nn.init.zeros_(submodule.bias)

    def insert_layers(self, layer: Module, name: str, bottom_name: str):
        """"""
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.insert_layer(layer, name, bottom_name)

    def replace_layers(self, layers: Union[List[Module], Module], names: Union[List[str], str]):
        """"""
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.replace_layer(layers, names)

    @property
    def path_model(self):
        """"""
        return self.root.joinpath(f"{self.model_name}.pth")

    @property
    def path_pred_target(self):
        """"""
        return self.root.joinpath(f"pred_target_{self.model_name}.csv")

    @property
    def path_loss_curve(self):
        """"""
        return self.root.joinpath(f"loss_curve_{self.model_name}.xlsx")

    def training(self):
        """"""
        print("\n----------------------------- START --------------------------------")
        # mkdir
        if not self.root.exists():
            self.root.mkdir(parents=True)

        if not self.k_fold:
            self.train_loop()
            self.save_pred_target()
            self.save_loss_curve()
        else:
            self.__kFold_data = self.k_fold_split
            for k in range(self.k_fold.get_n_splits()):
                self.train_loop(k)
                self.save_pred_target(k)
                self.save_loss_curve(k)

        self.save_total_r2()

    def train_loop(self, k=None):
        """"""
        # Setting training arguments
        self.init_model()  # Initialize model's arguments
        self.config_grad()  # Determine which layers require gradient calculation
        self.put_model_into_device()
        self.config_optimizer_scheduler()

        if self.auto_split:
            self.split_dataset()  # Split dataset
        elif not self.train_dataset:
            raise AttributeError(f"the train_dataset is {self.train_dataset}")

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if self.data_sampler else True,  # If sampler is specified, shuffle must not be specified.
            pin_memory=True,
            sampler=self.data_sampler(self.train_dataset) if self.data_sampler else None
        )

        for epoch in range(self.max_epoch):
            self.M.train()
            train_loss = []
            for batch in iter(loader):

                # Put batch into correct device
                batch = self.put_batch_into_device(batch)

                # Forward Propagation
                pred = self.model_forward(batch)

                # Calculate Loss
                target = batch.y.float()
                loss = self.loss_fn(pred, target)

                # Backpropagation
                self.optimizer_tool.backpropagation(loss)

                train_loss.append(loss)

            train_loss = torch.mean(torch.stack(train_loss).flatten())
            _, _, test_loss, r2 = self.test_loop()
            self.loss_curve.append(torch.stack([train_loss, test_loss, r2]).cpu().detach().numpy())

            # Print results
            t = time.localtime()
            print(f"\n{str(k) + 'th verify' if k is not None else ''} "
                  f"{t.tm_year}-{t.tm_mon}-{t.tm_mday} {t.tm_hour}:"
                  f"{t.tm_min}:{t.tm_sec} epoch{epoch}: train_loss: {train_loss}, test_loss:{test_loss}, r2: {r2}")

            # Adjust learning rate
            self.optimizer_tool.scheduler_step()

            # Save total model and state_dict
            if not self.mContainer.import_path or\
                    self.mContainer.import_path.parent != self.root:
                torch.save(self.M, self.root.joinpath(f"model_k{k}.pth" if k is not None else "model.pth"))
                torch.save(
                    self.M.state_dict(),
                    self.root.joinpath(f"state_dict_k{k}.pth" if k is not None else "state_dict.pth")
                )
            else:
                raise RuntimeError(f"DANGER: Overwriting the trained model!!!"
                                   f"path: {str(self.mContainer.import_path)}")

            # Change training args when training
            self.train_arg_change(epoch)

    def config_optimizer_scheduler(self):
        """"""
        self.optimizer_tool.config_optimizer(self.M)
        self.optimizer_tool.config_scheduler()

    def test_loop(self):
        """"""
        pred, target, _ = self.predict()
        loss = fn.mse_loss(pred, target)
        r2 = self.calculate_r2(pred, target)
        return pred, target, loss, r2

    def predict(self, all_data=False, **kwargs) -> (Tensor, Tensor, List):
        """"""
        self.M.train(False)
        loader = iter(
            DataLoader(
                self.dataset if all_data else self.test_dataset,
                batch_size=min(1024, len(self.dataset if all_data else self.test_dataset))
            )
        )
        list_pred = []
        list_target = []
        list_idt = []
        with torch.no_grad():
            for batch in loader:

                # Put batch into correct device
                batch = self.put_batch_into_device(batch)

                # Forward Propagation
                pred = self.model_forward(batch, **kwargs)
                target = batch.y.float()

                if self.save_id:
                    idt = batch.idt
                    list_idt.extend(idt)

                list_pred.append(pred)
                list_target.append(target)

        return torch.cat(list_pred), torch.cat(list_target), list_idt if self.save_id else None

    def save_pred_target(self, k=None):
        """"""
        pred, target, idx = self.predict()
        result = torch.stack([pred, target]).T.tolist()
        df = pd.DataFrame(result, columns=["pred", "target"], index=idx)
        if not k:
            df.to_csv(self.path_pred_target, mode="w", header=True)
        else:
            df.to_csv(self.path_pred_target, mode="a", header=False)

    def save_loss_curve(self, k=None):
        """"""
        self.loss_curve = np.stack(self.loss_curve)
        df = pd.DataFrame(self.loss_curve, columns=["train", "test", "r2"])
        with pd.ExcelWriter(self.path_loss_curve, engine="openpyxl", mode="w" if not k else "a") as writer:
            df.to_excel(writer, sheet_name=f"epoch {k}", header=True if not k else False)
        self.loss_curve = []  # Reset the attribute

    def save_total_r2(self):
        """    Calculate total R2 for all test_set     """
        df = pd.read_csv(self.path_pred_target, index_col=0)
        pred = torch.from_numpy(df.loc[:, "pred"].values)
        target = torch.from_numpy(df.loc[:, "target"].values)

        r2 = self.calculate_r2(pred, target).tolist()  # turn out gpu to cpu
        ser = pd.Series(r2, name="r2")
        ser.to_csv(self.root.joinpath("r2.csv"))

    @staticmethod
    def calculate_r2(pred: Tensor, target: Tensor) -> Tensor:
        """    Calculate r2 value for a given pred and target    """
        return 1 - torch.sum((pred - target) ** 2) / torch.sum((target - target.mean()) ** 2)

    def set_dataset(self, dataset):
        """"""
        self.dataset = dataset

    def setting(self, **kwargs):
        """"""
        """    Setting self attrs    """
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

    def set_grad_layers(self, grad_layers: Union[None, List[str], bool], exclude_grad_layers: List[str] = None):
        """"""
        self.mContainer.set_grad_layers(grad_layers, exclude_grad_layers)

    @property
    def k_fold_split(self):
        for k, (train_idx, test_idx) in enumerate(self.k_fold.split(self.dataset)):
            train_data, test_data = self.dataset[list(train_idx)], self.dataset[list(test_idx)]
            yield k, train_data, test_data

    def split_dataset(self):
        """
        split dataset into test set and train set, if self.k_fold is not None, perform k_
        :return: 2 lists: test set, train set
        """
        if self.k_fold:
            k, train_data, test_data = next(self.__kFold_data)
            print(f"{k}_fold dataset split!")
            self.train_dataset, self.test_dataset = train_data, test_data

        else:
            self.test_dataset, self.train_dataset = train_test_split(self.dataset, train_size=self.test_size)

    def load_model(self, import_model_path: Path = None):
        """"""
        if import_model_path:
            if import_model_path.parent == self.root:

                ImportWarning(
                    f"Danger: the import model is in the model file directory, path: {import_model_path}"
                    f"the import model might be overwrite when training."
                )

            self.mContainer = ModelContainer.load_model(import_model_path)

    def model_forward(self, batch, **kwargs):
        """"""
        pred = self.M(batch).flatten()
        return pred

    def put_batch_into_device(self, batch):
        """    Put batch into correct device    """
        if self.device_idx is None:
            batch.cpu()
        else:
            batch.cuda(self.device_idx)

        return batch

    def put_model_into_device(self):
        """"""
        # Choose device
        if self.device_idx is None:
            self.M.cpu()
        else:
            self.M.cuda(self.device_idx)

    @property
    def M(self) -> Module:
        """    Get training model    """
        return self.mContainer.M

    def print_para_grad(self):
        """"""
        for name, layers in self.M.named_parameters():
            print(f"{name}: {layers.requires_grad}")

    def train_arg_change(self, epoch: int):
        """"""
        if self.args_setter:
            self.args_setter(self, epoch)


class TrainModelT(object):
    """"""
    class EarlyStopCounter(object):
        """
        Count the times of epochs that the model has not been imported.
        When the times exceed the threshold, it returns a True, otherwise it returns a False
        """

        def __init__(self, stop_times=100):
            """"""
            self.stop_times = stop_times
            self.counter = self._counter()
            self.best_loss = None

        @staticmethod
        def _counter():
            """"""
            times = 0
            while True:
                yield times
                times += 1

        def reset_counter(self):
            self.counter = self._counter()

        def refresh(self):
            """"""
            self.best_loss = None

        def __call__(self, *args, **kwargs):
            """"""
            test_loss = kwargs["test_loss"]
            if self.best_loss is None or test_loss < self.best_loss:
                self.reset_counter()
                self.best_loss = test_loss

            count = next(self.counter)
            if count > self.stop_times:
                self.refresh()
                return True

            return False

    def __init__(self, root: Path, dataset=None, k_fold: int = None, model_name="model"):
        """"""
        self.root = root
        self.model_name = model_name
        self.data_slitter = DataSlitter(dataset, k_fold)
        self.max_epoch = 1000
        self.batch_size = 128
        self.loss_fn = torch.nn.MSELoss()
        self.device_idx = None
        self.mContainer = None
        self.optimizer_tool = OptimizerTool()
        self.save_id = False
        self.auto_split = False  # weather auto split dataset to train_set and text_set
        self.loss_curve = []
        self.is_transfer = False
        self.args_setter = None
        self.data_sampler: Union[None, type] = None
        self.stop_counter: Union["TrainModelT.EarlyStopCounter", None] = None

    def __repr__(self):
        """"""
        return f"----------------Train Neural Network Tool----------------\n" \
               f"\tModelName: {self.model_name}\n" \
               f"\tModelRoot: {self.root}\n" \
               f"\tDataset: {self.data_slitter}\n" \
               f"\tMaxEpoch: {self.max_epoch}\n" \
               f"\tBatchSize: {self.batch_size}\n" \
               f"\tDevice: {'cpu' if self.device_idx is None else 'cuda:' + str(self.device_idx)}\n" \
               f"\tTestSize: {self.data_slitter.test_size}\n" \
               f"\tTransferTrain: {self.is_transfer}\n" \
               f"\tSaveID: {self.save_id}\n" \
               f"\tModel: {self.M}\n"

    @property
    def early_stop_count(self):
        """"""
        if isinstance(self.stop_counter, self.EarlyStopCounter):
            return self.stop_counter.stop_times
        return None

    @early_stop_count.setter
    def early_stop_count(self, max_count=100):
        """"""
        if max_count:
            self.stop_counter = self.EarlyStopCounter(max_count)

    def early_stop(self, test_loss):
        """"""
        if isinstance(self.stop_counter, self.EarlyStopCounter):
            return self.stop_counter(test_loss=test_loss)
        return False

    @property
    def dataset(self) -> Dataset:
        """"""
        return self.data_slitter.dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        """"""
        self.data_slitter = DataSlitter(dataset, self.data_slitter.k_fold)

    @property
    def model(self) -> Union[Module, None]:
        """"""
        if self.mContainer is None:
            return None
        else:
            assert isinstance(self.mContainer, ModelContainer)
            return self.M

    @model.setter
    def model(self, model: Model):
        """"""
        self.mContainer = ModelContainer(model)

    @property
    def data_classifier(self):
        """"""
        return self.data_slitter.classifier

    @property
    def train_dataset(self):
        """"""
        return self.data_slitter.train_dataset

    @property
    def test_dataset(self):
        """"""
        return self.data_slitter.test_dataset

    @data_classifier.setter
    def data_classifier(self, classifier: Callable):
        """"""
        self.data_slitter.classifier = classifier

    def append_layers(self, layers: Union[List[Module], Module], names: Union[List[str], str]):
        """    Append new module layer after current model    """
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.append_layer(layers, names)

    def config_grad(self):
        """    Determine which layers require gradient calculation    """
        self.mContainer.config_grad()

    def init_model(self):
        """"""
        self.mContainer.reassemble_model()
        self.init_model_arg()

    def set_opti_args(self, opti_type: T = None, lr: float = None, **kwargs):
        """"""
        self.optimizer_tool.set_opti_args(opti_type, lr, **kwargs)

    def set_sch_args(self, sch_type: T, **kwargs):
        """"""
        self.optimizer_tool.set_sch_args(sch_type, **kwargs)

    @staticmethod
    def _k_fold(k_fold: int) -> Union[None, KFold]:
        """"""
        if k_fold is None:
            return None
        elif isinstance(k_fold, int) and k_fold > 1:
            return KFold(k_fold, shuffle=True)
        else:
            raise ValueError("the arg k_fold is required to be integer more than 1")

    def init_model_arg(self):
        """    initialize model's arguments    """
        if self.is_transfer:
            return

        for submodule in self.M.modules():
            if hasattr(submodule, "weight") and submodule.weight is not None:
                torch.nn.init.orthogonal_(submodule.weight)
            if hasattr(submodule, "bias") and submodule.bias is not None:
                torch.nn.init.zeros_(submodule.bias)

    def insert_layers(self, layer: Module, name: str, bottom_name: str):
        """"""
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.insert_layer(layer, name, bottom_name)

    def replace_layers(self, layers: Union[List[Module], Module], names: Union[List[str], str]):
        """"""
        assert isinstance(self.mContainer, ModelContainer)
        self.mContainer.replace_layer(layers, names)

    @property
    def path_model(self):
        """"""
        return self.root.joinpath(f"{self.model_name}.pth")

    @property
    def path_pred_target(self):
        """"""
        return self.root.joinpath(f"pred_target_{self.model_name}.csv")

    @property
    def path_loss_curve(self):
        """"""
        return self.root.joinpath(f"loss_curve_{self.model_name}.xlsx")

    @property
    def k_fold(self):
        """"""
        return self.data_slitter.k_fold

    def training(self):
        """"""
        print("\n----------------------------- START --------------------------------")
        # mkdir
        if not self.root.exists():
            self.root.mkdir(parents=True)

        if not self.k_fold:
            self.train_loop()
            self.save_pred_target()
            self.save_loss_curve()
        else:
            self.k_fold_split()
            for k in range(self.data_slitter.k):
                self.train_loop(k)
                self.save_pred_target(k)
                self.save_loss_curve(k)

        self.save_total_r2()

    def train_loop(self, k=None):
        """"""
        # Setting training arguments
        self.init_model()  # Initialize model's arguments
        self.config_grad()  # Determine which layers require gradient calculation
        self.put_model_into_device()
        self.config_optimizer_scheduler()

        if self.auto_split:
            self.split_dataset()  # Split dataset
        elif not self.train_dataset:
            raise AttributeError(f"the train_dataset is {self.train_dataset}")

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if self.data_sampler else True,  # If sampler is specified, shuffle must not be specified.
            pin_memory=True,
            sampler=self.data_sampler(self.train_dataset) if self.data_sampler else None
        )
        for epoch in range(self.max_epoch):
            self.M.train()
            train_loss = []
            for batch in iter(loader):

                # Put batch into correct device
                batch = self.put_batch_into_device(batch)

                # Forward Propagation
                pred = self.model_forward(batch)

                # Calculate Loss
                target = batch.y.float()
                loss = self.loss_fn(pred, target)

                # Backpropagation
                self.optimizer_tool.backpropagation(loss)

                train_loss.append(loss)

            train_loss = torch.mean(torch.stack(train_loss).flatten())
            _, _, test_loss, r2 = self.test_loop()
            self.loss_curve.append(torch.stack([train_loss, test_loss, r2]).cpu().detach().numpy())

            # Print training results
            self._print_result(k, epoch, train_loss.tolist(), test_loss.tolist(), r2.tolist())

            # Adjust learning rate
            self.optimizer_tool.scheduler_step()

            # Save total model and state_dict
            self.save_model(k, test_loss)

            # Change training args when training
            self.train_arg_change(k=k, epoch=epoch, train_loss=train_loss, test_loss=test_loss, r2=r2)

            # Whether to early stop
            if self.early_stop(test_loss):
                print(f"early stop!! count:{self.stop_counter.stop_times}")
                return

    def config_optimizer_scheduler(self):
        """"""
        self.optimizer_tool.config_optimizer(self.M)
        self.optimizer_tool.config_scheduler()

    def test_loop(self):
        """"""
        pred, target, _ = self.predict()
        loss = fn.mse_loss(pred, target)
        r2 = self.calculate_r2(pred, target)
        return pred, target, loss, r2

    def predict(self, all_data=False, **kwargs) -> (Tensor, Tensor, List):
        """"""
        self.M.train(False)
        loader = iter(
            DataLoader(
                self.data_slitter.dataset if all_data else self.test_dataset,
                batch_size=min(1024, len(self.data_slitter.dataset if all_data else self.test_dataset))
            )
        )
        list_pred = []
        list_target = []
        list_idt = []
        with torch.no_grad():
            for batch in loader:

                # Put batch into correct device
                batch = self.put_batch_into_device(batch)

                # Forward Propagation
                pred = self.model_forward(batch, **kwargs)
                target = batch.y.float()

                if self.save_id:
                    idt = batch.idt
                    list_idt.extend(idt)

                list_pred.append(pred)
                list_target.append(target)

        return torch.cat(list_pred), torch.cat(list_target), list_idt if self.save_id else None

    @property
    def save_best_model(self):
        """"""
        return self.mContainer.save_best_model

    @save_best_model.setter
    def save_best_model(self, best_one: bool):
        """"""
        assert isinstance(best_one, bool)
        self.mContainer.save_best_model = best_one

    def save_pred_target(self, k=None):
        """"""
        self.mContainer.load_current_model()
        pred, target, idx = self.predict()
        result = torch.stack([pred, target]).T.tolist()
        df = pd.DataFrame(result, columns=["pred", "target"], index=idx)
        if not k:
            df.to_csv(self.path_pred_target, mode="w", header=True)
        else:
            df.to_csv(self.path_pred_target, mode="a", header=False)

    def save_loss_curve(self, k=None):
        """"""
        self.loss_curve = np.stack(self.loss_curve)
        df = pd.DataFrame(self.loss_curve, columns=["train", "test", "r2"])
        with pd.ExcelWriter(self.path_loss_curve, engine="openpyxl", mode="w" if not k else "a") as writer:
            df.to_excel(writer, sheet_name=f"epoch {k}", header=True if not k else False)
        self.loss_curve = []  # Reset the attribute

    def save_total_r2(self):
        """    Calculate total R2 for all test_set     """
        df = pd.read_csv(self.path_pred_target, index_col=0)
        pred = torch.from_numpy(df.loc[:, "pred"].values)
        target = torch.from_numpy(df.loc[:, "target"].values)

        r2 = self.calculate_r2(pred, target).tolist()  # turn out gpu to cpu
        ser = pd.Series(r2, name="r2")
        ser.to_csv(self.root.joinpath("r2.csv"))

    def save_model(self, k: int, test_loss: Union[int, Tensor, List]) -> None:
        """
        Save model and model state dict onto dist
        Args:
            test_loss:
            k: k fold count

        Returns:
            None
        """
        self.mContainer.save_model(self.root, k)

    @staticmethod
    def calculate_r2(pred: Tensor, target: Tensor) -> Tensor:
        """    Calculate r2 value for a given pred and target    """
        return 1 - torch.sum((pred - target) ** 2) / torch.sum((target - target.mean()) ** 2)

    def set_dataset(self, dataset):
        """"""
        self.data_slitter = dataset

    def setting(self, **kwargs):
        """"""
        """    Setting self attrs    """
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

    def set_grad_layers(self, grad_layers: Union[None, List[str], bool], exclude_grad_layers: List[str] = None):
        """"""
        self.mContainer.set_grad_layers(grad_layers, exclude_grad_layers)

    def k_fold_split(self):
        """    get a generator to generate test_dataset"""
        self.data_slitter.k_fold_split()

    def split_dataset(self):
        """
        split dataset into test set and train set, if self.k_fold is not None, perform k_
        :return: 2 lists: test set, train set
        """
        self.data_slitter.split_dataset()

    def load_model(self, import_model_path: Path = None):
        """"""
        if import_model_path:
            if import_model_path.parent == self.root:

                ImportWarning(
                    f"Danger: the import model is in the model file directory, path: {import_model_path}"
                    f"the import model might be overwrite when training."
                )

            self.mContainer = ModelContainer.load_model(import_model_path)

    def load_trained_model(self):
        """"""
        self.mContainer.load_current_model()

    def model_forward(self, batch, **kwargs):
        """"""
        pred = self.M(batch).flatten()
        return pred

    def put_batch_into_device(self, batch):
        """    Put batch into correct device    """
        if self.device_idx is None:
            batch.cpu()
        else:
            batch.cuda(self.device_idx)

        return batch

    def put_model_into_device(self):
        """"""
        # Choose device
        if self.device_idx is None:
            self.M.cpu()
        else:
            self.M.cuda(self.device_idx)

    @property
    def M(self) -> Module:
        """    Get training model    """
        return self.mContainer.M

    @staticmethod
    def _print_result(k: int, epoch: int, train_loss: float, test_loss: float, r2: float):
        t = time.localtime()
        print(f"\n{str(k) + 'th verify' if k is not None else ''} "
              f"{t.tm_year}-{t.tm_mon}-{t.tm_mday} {t.tm_hour}:"
              f"{t.tm_min}:{t.tm_sec} epoch{epoch}: train_loss: {train_loss}, test_loss:{test_loss}, r2: {r2}")

    def print_para_grad(self):
        """"""
        for name, layers in self.M.named_parameters():
            print(f"{name}: {layers.requires_grad}")

    def train_arg_change(self, *args, **kwargs):
        """"""
        if self.args_setter:
            have_changed = self.args_setter(self, *args, **kwargs)
            if have_changed and self.stop_counter:
                self.stop_counter.reset_counter()


class TrainMultiBond(TrainModelT):
    """"""
    def __init__(self, root: Path, dataset=None, k_fold: int = None, model_name="model"):
        """"""
        super(TrainMultiBond, self).__init__(root, dataset, k_fold, model_name)
        self.bond_idx = []

    def model_forward(self, batch, **kwargs):
        """"""
        pred, bond_idx, num_ligand = self.M(batch)
        to_pred = kwargs.get("to_pred", False)
        if to_pred:
            self.bond_idx.append((bond_idx, num_ligand))
        return pred

    def save_pred_target(self, k=None):
        """"""
        pred, target, idx = self.predict(to_pred=True)
        result = torch.stack([pred, target]).squeeze().T.tolist()
        bond_idx = []
        first_idx = 0  # Recording the first index
        for bi, nl in self.bond_idx:  # bond_index, number of ligands
            bi[:, 0] = bi[:, 0] + first_idx
            bond_idx.append(bi)
            first_idx = first_idx + nl
        bond_idx = np.concatenate(bond_idx)
        idx = [[idx[i], c] for i, c in bond_idx]
        df = pd.DataFrame(result, columns=["pred", "target"], index=np.array(idx).T.tolist())
        if not k:
            df.to_csv(self.path_pred_target, mode="w", header=True)
        else:
            df.to_csv(self.path_pred_target, mode="a", header=False)


class ConditionSetter(object):
    """"""
    def __init__(self, tm: TrainModelT, determiner: Callable, setter: Callable, *setter_kwargs: Dict):
        """"""
        self.tm = tm
        self.setter_kwargs = iter(setter_kwargs)
        self.determiner = determiner
        self.setter = setter

    def __call__(self, *args, **kwargs) -> bool:
        """"""
        if self.determiner(*args, **kwargs):
            try:
                kwargs_ = next(self.setter_kwargs)
            except StopIteration:
                kwargs_ = None

            if kwargs_ is not None:
                self.setter(self.tm, **kwargs_)
                return True

        return False

    def re_init(self):
        """"""
        return self.__class__(self.tm, self.determiner, self.setter, *list(self.setter_kwargs))


def grad_setter(tm: TrainModelT, **kwargs):
    """"""
    tm.set_grad_layers(**kwargs)
    tm.config_grad()
    tm.print_para_grad()


class ChangeKwargsName(object):
    """"""
    def __init__(self, name_transform: Dict):
        """"""
        self.name_transform = name_transform

    def __call__(self, func: Callable):
        """"""
        def wrapper(*args, **kwargs):
            """"""
            input_kwargs = {}
            for original_name, changed_name in self.name_transform.items():
                value = kwargs.get(original_name, None)
                if value is not None:
                    input_kwargs[changed_name] = value

            return func(**input_kwargs)

        return wrapper
