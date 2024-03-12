from multiparty_diarization.datasets.dataset_wrapper import DiarizationDataset
from xml.etree import ElementTree
import librosa
import xml.etree.ElementTree as ET
import torch

from typing import List, Tuple, Dict
from multiparty_diarization.datasets.ami import AMI
import os
import numpy as np

class AMIMultiChannel(AMI):

    """Class for AMI Dataset with Lavs"""

    def __init__(self, 
                array_root_path: str,
                lav_root_path: str,
                sample_len_s: float = 15.0,
                min_speaker_gap_s: float = 1.0 
                ):
        
        """Create instance of AMIMultiChannel dataset
        
        Args:
            array_root_path (str): path to AMI dataset directory with array audio
            lav_root_path (str): path to AMI dataset directory with lav audio
            sample_len_s (float): target length of each sample
            min_speaker_gap_s: minimum interval of non-speech to separate speaker segments
        """

        self.root_path = array_root_path
        self.array_root_path = array_root_path
        self.lav_root_path = lav_root_path
        self.sample_len_s = sample_len_s
        self.min_speaker_gap_s = min_speaker_gap_s
        self.sample_rate = 16000

        word_dir = os.path.join(array_root_path, 'words') # Contains transcript info
        meeting_IDs = [subdir for subdir in next(os.walk(array_root_path))[1] if subdir[0].isupper() and subdir != 'MultiChannel'] # List all meetings

        meeting_word_XML_files = { # List XML transcript files associated each meeting (one per participant)
            meeting_ID: [
                    os.path.join(word_dir, fname) for fname in os.listdir(word_dir) if fname.split('.')[0] == meeting_ID
                    ]
                for meeting_ID in meeting_IDs
            }

        # Generate samples
        self._generate_samples(meeting_word_XML_files)


    def _load_lav_audio(self, sample_info: dict):

        lav_path = os.path.join(self.lav_root_path, sample_info['meeting_ID'], 'audio')
        lav_files = os.listdir(lav_path)

        lav_channels = [
            librosa.load(os.path.join(lav_path, file), 
                offset = sample_info['start'], 
                duration = sample_info['end'] - sample_info['start'],
                sr = self.sample_rate
                )[0]
                for file in lav_files
            ]
        
        lav_audio = np.stack(lav_channels, 0)

        return lav_audio


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float, float]], dict]: 
        
        """Get sample from dataset by index
        
        Args:
            idx (int): index of desired samples

        Returns:
            sample (tuple): first element is a tuple with audio waveforms (array_audio, lav_audio)
            Second item is a list of true diarization intervals of the form (speaker, start, end). 
            Third item is a dict with additional sample info
        """

        sample, sample_info = self.samples[idx], self.sample_info[idx]

        sample = self._normalize_sample(sample, sample_info)

        # Load audio, assummes a single-channel beamformed file named 'beamform.wav'
        audio_path_array = os.path.join(self.array_root_path, sample_info['meeting_ID'], 'audio', "beamform.wav")
        
        array_audio, _ = librosa.load(audio_path_array, 
                offset = sample_info['start'], 
                duration = sample_info['end'] - sample_info['start'],
                sr = self.sample_rate
                )
        
        lav_audio = self._load_lav_audio(sample_info)

        array_audio = torch.from_numpy(array_audio).unsqueeze(0)
        lav_audio = torch.from_numpy(lav_audio)

        return (array_audio, lav_audio), sample, sample_info
                
if __name__ == "__main__":

    from random import randint
    import torchaudio

    arr_root_path = '/proj/disney/amicorpus_array'
    lav_root_path = '/proj/disney/amicorpus_lav/'
    dset = AMIMultiChannel(arr_root_path, lav_root_path)

    print('Dataset generation complete')

    audio, segments, info = dset[ randint(0, len(dset)) ]

    print(audio[0].shape, audio[1].shape)
    
    for segment in segments:
        print("SPEAKER %s <%.2f - %.2f>" % segment)


