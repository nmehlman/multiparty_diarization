from multiparty_diarization.datasets.dataset_wrapper import DiarizationDataset
import json
import librosa
import torch

from typing import List, Tuple, Dict
import os

class DinnerParty(DiarizationDataset):

    """Class for Dinner Party Dataset"""

    def __init__(self, 
                root_path: str, 
                split: str = 'eval',
                sample_len_s: float = 15.0,
                min_speaker_gap_s: float = 1.0 
                ):
        
        """Create instance of Dinner Party dataset
        
        Args:
            root_path (str): path to Dinner Party dataset directory
            sample_len_s (float): target length of each sample
            min_speaker_gap_s: minimum interval of non-speech to separate speaker segments
        """

        self.audio_dir = os.path.join(root_path, 'audio', split)
        self.transcript_dir = os.path.join(root_path, 'transcripts', split)

    
    def _generate_samples(self):

        self.samples = []
        self.sample_info = []

        for session_file in os.listdir()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float, float]], dict]: 
        
        sample, sample_info = self.samples[idx], self.sample_info[idx]

        sample = self._normalize_sample(sample, sample_info)

        # TODO add multi-channel support
        # Load audio, assummes a single-channel beamformed file named 'beamform.wav'
        audio_path = os.path.join(self.root_path, sample_info['meeting_ID'], 'audio', "beamform.wav")
        
        audio, _ = librosa.load(audio_path, 
                offset = sample_info['start'], 
                duration = sample_info['end'] - sample_info['start'],
                sr = 16000
                )

        audio = torch.from_numpy(audio).unsqueeze(0)

        return audio, sample, sample_info
                
if __name__ == "__main__":

    from random import randint
    import torchaudio

    root_path = '/proj/disney/Dinner Partycorpus'
    dset = DinnerParty(root_path=root_path)

    print('Dataset generation complete')

    audio, segments = dset[ randint(0, len(dset)) ]
    
    for segment in segments:
        print("SPEAKER %s <%.2f - %.2f>" % segment)

    torchaudio.save('sample.wav', audio, 16000)



