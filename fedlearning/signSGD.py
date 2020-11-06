import torch

# My libraries
from fedlearning import Compressor

class SignSGDCompressor(Compressor):

    def __init__(self, config):
        super().__init__()

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encoded_tensor = (tensor >= 0).to(torch.float)
        return encoded_tensor

    def decompress(self, tensor):
        """Decode the signs to float format """
        decoded_tensor = tensor * 2 - 1
        return decoded_tensor

    def normalize_aggregation(self, tensor, threshold):
        normalized_tensor = (tensor >= threshold).to(torch.float) * 2 - 1
        return normalized_tensor
    
class StoSignSGDCompressor(Compressor):
    
    def __init__(self, config):
        super().__init__()
        self.b = config.b

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        random_variable = torch.rand_like(tensor)
        ones_tensor = torch.ones_like(tensor)
        zeros_tensor = torch.zeros_like(tensor)
        encoded_tensor = torch.where(random_variable<=(1/2+tensor/self.b), ones_tensor, zeros_tensor)
        return encoded_tensor

    def decompress(self, tensor):
        """Decode the signs to float format """
        decoded_tensor = tensor * 2 - 1
        return decoded_tensor

    def normalize_aggregation(self, tensor, threshold):
        normalized_tensor = (tensor >= threshold).to(torch.float) * 2 - 1
        return normalized_tensor
    