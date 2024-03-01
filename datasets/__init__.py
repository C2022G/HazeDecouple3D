from .nerf import NeRFDataset
from .llff import LLFFDataset

dataset_dict = {'nerf': NeRFDataset,
                'llff': LLFFDataset}
