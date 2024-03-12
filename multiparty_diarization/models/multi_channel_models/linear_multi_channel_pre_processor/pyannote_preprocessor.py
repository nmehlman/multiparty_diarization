from pyannote.audio.pipelines.utils import get_model
import torch
from pyannote.audio.models.segmentation import PyanNet
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from multiparty_diarization.multi_channel_models.linear_multi_channel_pre_processor.preprocessor import MultiChanLinearPreprocessor
from multiparty_diarization.multi_channel_models.pyannote_multi_channel.utils import MutiChannelAudio
import pdb

class PyanNetPreprocessor(PyanNet):
   
    """Modified version of PyanNet Pyannote model that supports multi-channel 
    via linear preprocessor from https://arxiv.org/abs/2110.04694"""

    def __init__(self, 
                num_channels: int,
                preprocessor_kwargs = {},
                lr = 1e-3,
                *args, 
                **kwargs):

        """Initialized PyanNetMultiChannel instance
    
        Argss:

            num_channels (int): Number of channels.
            
            sincnet (dict, optional): Keyword arugments passed to the SincNet block.
                Defaults to {"stride": 1}.

            lstm (dict, optional): Keyword arguments passed to the LSTM layer.
                Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
                i.e. two bidirectional layers with 128 units each.
                Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
                This may proove useful for probing LSTM internals.

            linear (dict, optional): Keyword arugments used to initialize linear layers
                Defaults to {"hidden_size": 128, "num_layers": 2},
                i.e. two linear layers with 128 units each.
        """

        super().__init__(*args, **kwargs)

        self.single_channel_model = get_model("pyannote/segmentation@2022.07")
        self.single_channel_model.freeze()

        self.specs = self.single_channel_model.specifications
        self.audio = MutiChannelAudio( sample_rate = kwargs.get('sample_rate', 16000) )

        self.lr = lr

        self.preprocessor = MultiChanLinearPreprocessor(num_channels=num_channels, **preprocessor_kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Args:
            waveforms (torch.tensor): Tensor representing audio waveforms of shape (batch, channel, sample)

        Returns:
            scores (torch.tensor): Tensor of scores with shape (batch, frame, classes)
        """

        mixing_coeff = self.preprocessor(waveforms)

        waveforms_mix = torch.sum( waveforms * mixing_coeff, 1, keepdim=True )

        scores = self.single_channel_model(waveforms_mix)

        return scores
