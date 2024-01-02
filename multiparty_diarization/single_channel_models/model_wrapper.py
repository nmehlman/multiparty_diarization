import torch 
from typing import List, Tuple

class DiarizationModel:

    """Wrapper class for all diarization models to support unifrom interface"""

    def __call__(self, waveform: torch.tensor, **kwargs) -> List[Tuple[str, float, float]]:

        """Applies diariazation to audio signal
        
        Args:
            waveform (torch.tensor): audio array of shape (channels, samples)
            kwargs (dict, optional): additional kwargs to support model-specific needs

        Returns:
            results (list): list of tuples for each segment of the form (speaker, start, end)
        """

        raise NotImplementedError

