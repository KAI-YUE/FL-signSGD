import numpy as np
import copy
from collections import OrderedDict

# Pytorch Libraries
import torch

# My Libraries 
from deeplearning.networks import init_weights

def generate_attacker_list(config):
    user_ids = np.arange(config.users)
    np.random.shuffle(user_ids)
    attacker_list = user_ids[:config.num_attackers]
    normal_user_ids = user_ids[config.num_attackers:]

    return dict(attacker_list=attacker_list, 
                normal_user_ids=normal_user_ids)

def random_grad_package(model):
    model_state_dict = model.state_dict()
    random_grad = OrderedDict()

    for w_name, w_value in model_state_dict.items():
        random_grad[w_name + ".grad"] = rand_variable = torch.randint_like(w_value, high=1)
    
    return random_grad
