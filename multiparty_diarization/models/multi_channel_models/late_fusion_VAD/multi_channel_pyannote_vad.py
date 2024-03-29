from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection
import torchaudio
from functools import partial
from typing import Callable, Optional
import numpy as np
from pyannote.audio.core.io import AudioFile

class MultiChannelVAD(VoiceActivityDetection):

    def __init__(self, 
                    *args,
                    **kwargs):

        super().__init__(*args, **kwargs)
        self._segmentation.skip_aggregation = True
        self.instantiate(self.default_parameters())

    def apply(self, file: AudioFile, hook: Optional[Callable] = None):
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, hook=partial(hook, "segmentation", None)
                )
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            
            segmentations = []
            for i in range(file['waveform'].shape[0]): # Apply VAD separatley to each channel

                channel_file = {
                                "waveform": file['waveform'][i, :].reshape(1,-1), 
                                "sample_rate": file['sample_rate']
                            }

                segmentations.append( self._segmentation(
                    channel_file, hook=partial(hook, "segmentation", None)
                    ) 
                )

            segmentations = np.stack(segmentations, axis=0)

        # Take max over diarization model's speaker channels
        segmentations = np.max(segmentations, axis=-1, keepdims=False)
        segmentations = np.transpose(segmentations, (1,2,0))

        return segmentations

    
if __name__ == "__main__":

    import torch
    import torchaudio
    import os

    HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

    model = MultiChannelVAD(num_channels=4)
    waveform, sr = torchaudio.load('/home/nmehlman/disney/multiparty_diarization/misc/test_audio.wav')
    waveform = torch.concat([waveform, waveform, waveform, waveform], 0)
    segs = model({"waveform": waveform, "sample_rate": sr})
    print(segs.data.shape)
