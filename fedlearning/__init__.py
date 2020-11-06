from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self):
        self._require_grad_idx = False

    @abstractmethod
    def compress(self, tensor):
        """Compresses a tensor with the given compression context, and then returns it with the context needed to decompress it."""

    @abstractmethod
    def decompress(self, tensors):
        """Decompress the tensor with the given decompression context."""

    @abstractmethod
    def normalize_aggregation(self, tensors):
        """Normalize the aggregated results"""

from fedlearning.signSGD import *

compressor_registry = {
"signSGD": SignSGDCompressor,
"sto_signSGD": StoSignSGDCompressor
}
