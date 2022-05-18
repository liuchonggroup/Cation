"""
python v3.7.9
@Project: 1-2.split_crystals_to_pairs.py
@File   : training_aids.py
@Author : Zhiyuan Zhang
@Date   : 2022/4/26
@Time   : 19:22
"""
from typing import Union, List, Dict, TypeVar, Tuple, Callable
from pathlib import Path
import copy
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss

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
        self.train_model = copy.deepcopy(Model(self.model, self.append_layers, self.replace_layers, self.insert_layers))

    def set_grad_layers(self, grad_layers: Union[None, List[str], bool], exclude_grad_layers: List[str] = None):
        """    Assign layers are allowed to calculate gradient when training    """
        self._grad_layers = grad_layers
        self._exclude_grad_layers = exclude_grad_layers if exclude_grad_layers else []

    def update_model(self) -> None:
        """"""
        self.reassemble_model()


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


class ConditionSetter(object):
    """"""
    def __init__(self, determiner: Callable, setter: Callable, *setter_kwargs: Dict):
        """"""
        self.setter_kwargs = iter(setter_kwargs)
        self.determiner = determiner
        self.setter = setter

    def __call__(self, *args, **kwargs):
        """"""
        if self.determiner(*args, **kwargs):
            try:
                kwargs_ = next(self.setter_kwargs)
            except StopIteration:
                kwargs_ = False

            if kwargs_:
                self.setter(*args, **kwargs)
