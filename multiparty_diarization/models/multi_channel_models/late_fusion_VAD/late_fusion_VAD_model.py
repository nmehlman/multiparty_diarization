from multiparty_diarization.models.model_wrapper import DiarizationModel
from multiparty_diarization.models.multi_channel_models.late_fusion_VAD.vad_fusion_speaker_diarization import VADFusionSpeakerDiarization
import torch

class LateFusionVAD(DiarizationModel):

    def __init__(
            self, 
            vad_weight: float = 0.5,
            device: str = 'cpu'
        ):

        """Creates LateFusionVAD instance
        
        Args:
            vad_weight (float): weight to place on lav-channel VAD in diarization predictions
            device (str): device to run evaluation on (cpu or gpu)
        """

        super().__init__()

        self.device = device

        # Load pipeline
        self._model = VADFusionSpeakerDiarization(vad_weight=vad_weight)

        # Move to device (cpu or gpu)
        self._model = self._model.to(torch.device(device))

    def __call__(
                self, 
                audio: tuple,
                sample_rate: int, 
            ):

        """Applies diariazation to audio signal
        
        Args:
            audio (tuple): array_audio, lav_audio
            sample_rate (int): audio sampling rate

        Returns:
            results (list): list of tuples for each segment of the form (speaker, start, end)
        """

        array_waveform, lav_waveform = audio

        array_waveform = array_waveform.to(self.device)
        lav_waveform = lav_waveform.to(self.device)

        arr_audio_in_memory = {"waveform": array_waveform, "sample_rate": sample_rate}
        ct_audio_in_memory = {"waveform": lav_waveform, "sample_rate": sample_rate}

        diarization = self._model(arr_audio_in_memory, ct_audio_in_memory) # Call model

        results = [ (str(spkr), speech_turn.start, speech_turn.end)
            for speech_turn, spkr, _ in diarization.itertracks(yield_label=True)
            ]

        return results