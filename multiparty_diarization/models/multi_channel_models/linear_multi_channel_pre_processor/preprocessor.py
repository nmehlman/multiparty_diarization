import torch
import torch.nn as nn
import torchaudio

class MultiChanLinearPreprocessor(nn.Module):

    def __init__(self, 
                 num_channels: int, 
                 sample_rate: int = 16000,
                 spectrogram_kwargs: dict = {}
                 ):

        super().__init__()

        self.num_channels = num_channels

        # Output shape: (B, C, F, T)
        self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **spectrogram_kwargs)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4,4))
        self.tanh = nn.Tanh()

        self.attention = nn.MultiheadAttention(embed_dim=125, num_heads=5)

        self.projector = nn.Linear(in_features=125, out_features=1)

    def forward(self, x: torch.Tensor):

        T = x.shape[-1] # Number of time-steps

        # Check channel count
        assert x.shape[1] == self.num_channels, "incorrect number of input channels"

        x_spec = self.spectrogram(x)
        z = self.tanh( self.conv1(x_spec) ) # (B,C,F,T')

        z_cc_attention = []
        for t in range(z.shape[-1]): # Apply cross-channel attention per-time step
            zt, _ = self.attention(z[:,:,:,t], z[:,:,:,t], z[:,:,:,t])
            z_cc_attention.append(zt)

        z = torch.stack(z_cc_attention, -1) # (B,C,F,T')
        z = z.transpose(-1, -2) # (B,C,T',F)
        z = self.projector(z).squeeze(-1) # (B,C,T') # Project out frequency dimension

        # Interpolate back to original time grid
        z = nn.functional.interpolate(z, T, mode='linear')

        # Apply softmax per time step
        z = nn.functional.softmax(z, dim=1)

        return z
    
if __name__ == "__main__":

    model = MultiChanLinearPreprocessor(num_channels=8)

    x = torch.randn(16,8,16000)
    z = model(x)

    print(z.sum(1))