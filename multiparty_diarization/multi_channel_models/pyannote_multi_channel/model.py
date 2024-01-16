from pyannote.audio.pipelines.utils import get_model
import torch
from pyannote.audio.models.segmentation import PyanNet
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from multiparty_diarization.multi_channel_models.pyannote_multi_channel.utils import MutiChannelAudio
import pdb

class PyanNetMultiChannel(PyanNet):
   
    """Modified version of PyanNet Pyannote model that supports multi-channel 
    via cross channel and cross frame attention from https://arxiv.org/abs/2110.04694"""

    def __init__(self, 
                attention_heads_cc: int = 4, 
                attention_heads_cf: int = 4, 
                use_pretrained_weights: bool = False,
                freeze_encoders: bool = False,
                lr: float = 1e-3,
                *args, 
                **kwargs):

        """Initialized PyanNetMultiChannel instance
    
        Args:

            attention_heads_cc (int, optional): Number of cross channel attention heads. Defaults to 4

            attention_heads_cf (int, optional): Number of cross frame attention heads. Defaults to 4 

            use_pretrained_weights (bool, optional): Use encoder/linear weights from pre-trained model as starting point.
                Defaults to False

            freeze_encoders (bool, optional): Freeze weights of encoder module. Defaults to False

            lr (float, optional): Learning rate. Defaults to 1e-3
            
            sample_rate (int, optional): Audio sample rate. Defaults to 16000

            num_channels (int, optional): Number of channels. Defaults to mono (1)
            
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

        _model = get_model("pyannote/segmentation@2022.07")
        self.specs = _model.specifications
        self.audio = MutiChannelAudio( sample_rate = kwargs.get('sample_rate', 16000) )

        self.lr = lr

        if use_pretrained_weights: # Copy weights from pretrained model
            self.sincnet = _model.sincnet
            self.lstm = _model.lstm
            self.linear = _model.linear
            self.activation = _model.activation
            self.classifier = _model.classifier

        D = _model.hparams['lstm']['hidden_size'] * 2
        
        self.attention_cc = nn.MultiheadAttention(embed_dim=D, num_heads=attention_heads_cc, batch_first=True)
        self.layer_norm_cc = nn.Identity()#nn.LayerNorm(D)

        self.attention_cf = nn.MultiheadAttention(embed_dim=D, num_heads=attention_heads_cf, batch_first=True)
        self.layer_norm_cf = nn.Identity()#nn.LayerNorm(D)

        self.freeze_encoders = freeze_encoders

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Args:
            waveforms (torch.tensor): Tensor representing audio waveforms of shape (batch, channel, sample)

        Returns:
            scores (torch.tensor): Tensor of scores with shape (batch, frame, classes)
        """

        if self.freeze_encoders:
            with torch.no_grad():
                outputs = [self.sincnet(waveforms[:,chan,:].unsqueeze(1)) for chan in range(waveforms.size(1))]

                if self.hparams.lstm["monolithic"]:
                    outputs = [self.lstm(
                        rearrange(output, "batch feature frame -> batch frame feature")
                    )[0] for output in outputs]
                else:
                    
                    outputs = [rearrange(output, "batch feature frame -> batch frame feature")
                            for output in outputs]
                    
                    for i, lstm in enumerate(self.lstm):
                        outputs = [lstm(output)[0] for output in outputs]
                        if i + 1 < self.hparams.lstm["num_layers"]:
                            outputs = [self.dropout(output) for output in outputs]

        else:
            
            outputs = [self.sincnet(waveforms[:,chan,:].unsqueeze(1)) for chan in range(waveforms.size(1))]

            if self.hparams.lstm["monolithic"]:
                outputs = [self.lstm(
                    rearrange(output, "batch feature frame -> batch frame feature")
                )[0] for output in outputs]
            else:
                
                outputs = [rearrange(output, "batch feature frame -> batch frame feature")
                        for output in outputs]
                
                for i, lstm in enumerate(self.lstm):
                    outputs = [lstm(output)[0] for output in outputs]
                    if i + 1 < self.hparams.lstm["num_layers"]:
                        outputs = [self.dropout(output) for output in outputs]
        
        # Cross channel attention (based on https://arxiv.org/abs/2110.04694)          
        outputs = torch.stack(outputs,1) # B x C x T x D

        T = outputs.shape[2]
        outputs_attn_cc = [
            self.layer_norm_cc( 
                    outputs[:, :, t, :] + self.attention_cc(outputs[:, :, t, :], outputs[:, :, t, :], outputs[:, :, t, :])[0] 
                )
                for t in range(T)
        ]

        outputs_attn_cc = torch.stack(outputs_attn_cc, 2)

        # Cross frame attention (based on https://arxiv.org/abs/2110.04694)
        C = outputs_attn_cc.shape[1]
        outputs_attn_cf = [
            self.layer_norm_cf( 
                    outputs_attn_cc[:, c, :, :] + self.attention_cc(outputs_attn_cc[:, c, :, :], outputs_attn_cc[:, c, :, :], outputs_attn_cc[:, c, :, :])[0] 
                )
                for c in range(C)
        ]

        outputs_attn_cf = torch.stack(outputs_attn_cf, 1)
        outputs = outputs_attn_cf.mean(1) # Average over channels

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))

if __name__ == "__main__":

    model = PyanNetMultiChannel()
    
    print(model.audio)