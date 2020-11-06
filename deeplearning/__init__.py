from .dataset import UserDataset, assign_user_data
from .networks import *

nn_registry = {
"naiveMLP": NaiveMLP,
"naiveCNN": NaiveCNN,

"bn": FullPrecision_BN
}
