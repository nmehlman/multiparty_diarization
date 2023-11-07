from model_wrapper import DiarizationModel
from pyannote.audio import Pipeline
import os
import torch

# Get token required for model acess
HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

class PyannoteDiarization(DiarizationModel):

    def __init__(
            self, 
            model_name: str = "pyannote/speaker-diarization-3.0",
            device: str = 'cpu'
        ):

        """Creates PyannoteDiarization instance
        
        Args:
            model_name (str): name of pretrained Pyannote model to use
            device (str): device to run evaluation on (cpu or gpu)
        """

        super().__init__()

        self.device = device

        # Load pipeline
        self._model = Pipeline.from_pretrained(
            model_name,
            use_auth_token=HUGGING_FACE_TOKEN)

        # Move to device (cpu or gpu)
        self._model = self._model.to(torch.device(device))

    def __call__(self, waveform: torch.tensor, sample_rate: int):

        """Applies diariazation to audio signal
        
        Args:
            waveform (torch.tensor): audio array of shape (channels, samples)
            sample_rate (int): audio sampling rate

        Returns:
            results (list): list of tuples for each segment of the form (speaker, start, end)
        """

        assert waveform.shape[0] == 1, 'only single channel audio supported'

        #waveform = waveform.to(self.device)

        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self._model(audio_in_memory)

        results = [ (str(spkr), speech_turn.start, speech_turn.end)
            for speech_turn, spkr, _ in diarization.itertracks(yield_label=True)
            ]

        return results

if __name__ == "__main__":

    import torchaudio

    waveform, fs = torchaudio.load('../../misc/test_audio.wav')

    model = PyannoteDiarization(device='cuda:0')
    
    results = model(waveform, fs)
    
    print('\n*** Diarization Results ***')
    for spkr, start, end in results:
        print(f'SPEAKER {spkr}: [{start:.2f} -->> {end:.2f}]')
