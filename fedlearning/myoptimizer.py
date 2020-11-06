import logging
import copy
from collections import OrderedDict

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from deeplearning import UserDataset
from config.utils import parse_dataset_type
from fedlearning import compressor_registry
from fedlearning.buffer import GradBuffer

class LocalUpdater(object):
    def __init__(self, user_resource, config, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarry):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - device (str):     set 'cuda' or 'cpu' for the user. 
                - predictor (str):  predictor type.
                - quantizer (str):  quantizer type.
        """
        
        try:
            self.lr = user_resource["lr"]
            self.momentum = user_resource["momentum"]
            self.weight_decay = user_resource["weight_decay"]
            self.batch_size = user_resource["batch_size"]
            self.device = user_resource["device"]
            
            assert("images" in user_resource)
            assert("labels" in user_resource)
        except KeyError:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batch_size`!") 
        except AssertionError:
            logging.error("LocalUpdater Initialization Failure! Input should include samples!") 

        dataset_type = parse_dataset_type(config)

        if config.imbalance:
            sampler = WeightedRandomSampler(user_resource["sampling_weight"], 
                                    num_samples=user_resource["sampling_weight"].shape[0])

            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                                user_resource["labels"],
                                dataset_type), 
                            sampler=sampler,
                            # sampler=None,
                            batch_size=self.batch_size)
        else:
            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                            user_resource["labels"],
                            dataset_type), 
                    sampler=None, 
                    batch_size=self.batch_size,
                    shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.compressor = compressor_registry[config.compressor](config)
        self.compressed_grad = None

    def local_step(self, model):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model.
        """
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

        for sample in self.sample_loader:
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)
            optimizer.zero_grad()

            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()

            compressed_grad = self.compress_grad(model)
            optimizer.zero_grad()
            break
        
        self.compressed_grad = compressed_grad

    def compress_grad(self, model):
        compressed_grad = OrderedDict()
        named_modules = model.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif module.weight is None:
                continue
            compressed_grad[module_name + ".weight.grad"] = self.compressor.compress(module.weight.grad)
            
            if module.bias is None:
                continue
            compressed_grad[module_name + ".bias.grad"] = self.compressor.compress(module.bias.grad)

        return compressed_grad

    def uplink_transmit(self):
        """Simulate the transmission of residual between local updated weight and local received initial weight.
        """ 
        return self.compressed_grad


class GlobalUpdater(object):
    def __init__(self, config, initial_model, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:

            initial_model (OrderedDict): initial model state_dict
        """
        self.num_users = config.users
        self.lr = config.lr

        self.accumulated_delta = None
        self.compressor = compressor_registry[config.compressor](config)

    def global_step(self, model, local_packages, **kwargs):
        """Perform a global update with collocted coded info from local users.
        """
        accumulated_grad = GradBuffer(local_packages[0], mode="zeros") 
        for i, package in enumerate(local_packages):
            accumulated_grad += GradBuffer(package)

        accumulated_grad.normalize(self.compressor, self.num_users/2)
        grad_dict = accumulated_grad.grad_dict()

        named_modules = model.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif module.weight is None:
                continue
            module.weight.data -= self.lr*grad_dict[module_name + ".weight.grad"]
            
            if module.bias is None:
                continue
            module.bias.data -= self.lr*grad_dict[module_name + ".bias.grad"]