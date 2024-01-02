from torch.utils.data import Dataset
from typing import Tuple, List
import torch

class DiarizationDataset(Dataset):

    """Wrapper class for all diarization datasets to support unifrom interface"""

    def __init__(self, *args, **kwargs):
        """Initialize dataset"""
        super().__init__()

    def __len__(self) -> int:
        """Computes number of samples in dataset"""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float, float]], dict]:
        """Get sample from dataset by index
        
        Args:
            idx (int): index of desired samples

        Returns:
            sample (tuple): first element is the audio waveform as a torch.Tensor with shape = (channels, samples)
            Second item is a list of true diarization intervals of the form (speaker, start, end). 
            Third item is a dict with additional sample info
        """
        raise NotImplementedError

    def generate_oracle_info(self, idx: int) -> dict:
        """Return oracle info for sample based on idx

         Args:
            idx (int): index of desired samples

        Returns:
            info (dict): information needed for oracle evaluation
        """
        
        raise NotImplementedError

