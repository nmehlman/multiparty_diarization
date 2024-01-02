from multiparty_diarization.datasets.dataset_wrapper import DiarizationDataset
import json
import librosa
import torch
import numpy as np

from typing import List, Tuple, Dict
import os

def time_to_seconds(time_str):
    """
    Convert a time string in the format HH:MM:SS.ss to seconds (float).

    Args:
    time_str (str): A time string in the format HH:MM:SS.ss

    Returns:
    float: The time in seconds.
    """
    hours, minutes, seconds = time_str.split(':')
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


class DinnerParty(DiarizationDataset):

    """Class for Dinner Party Dataset"""

    def __init__(self, 
                root_path: str, 
                split: str = 'eval',
                sample_len_s: float = 15.0,
                ):
        
        """Create instance of Dinner Party dataset
        
        Args:
            root_path (str): path to Dinner Party dataset directory
            sample_len_s (float): target length of each sample
        """

        self.sample_len_s = sample_len_s

        self.audio_dir = os.path.join(root_path, 'audio', split)
        self.transcript_dir = os.path.join(root_path, 'transcriptions', split)
        self._generate_samples()

    def _mean_start_time(self, utterance):
        """Untility function to average start time labels across devices"""
        return np.mean([time_to_seconds(time_str) for time_str in utterance['start_time'].values()])

    def _mean_end_time(self, utterance):
        """Untility function to average end time labels across devices"""
        return np.mean([time_to_seconds(time_str) for time_str in utterance['end_time'].values()])
    
    def _generate_samples(self):

        """Populates samples"""

        self.samples = []
        self.sample_info = []

        for session_file in os.listdir(self.transcript_dir): # For each session
    
            full_path = os.path.join(self.transcript_dir, session_file)
            session_data = json.load(open(full_path))

            # Sort utterances by start time
            session_data = sorted(session_data, key = self._mean_start_time)
            
            current_sample = None
            for utterance in session_data:
                
                spkr = utterance['speaker_id']
                session = utterance['session_id']
                utt_start = self._mean_start_time(utterance)
                utt_end = self._mean_end_time(utterance)

                if current_sample is None: # Initialize new sample
                    
                    sample_start = utt_start
                    current_sample_len = utt_end - utt_start
                    current_sample = [(spkr, 0.0, utt_end - sample_start)]

                else: # Continue sample
                    current_sample.append(((spkr, utt_start - sample_start, utt_end - sample_start)))
                    current_sample_len = utt_end - sample_start

                if current_sample_len >= self.sample_len_s: # Reached desired duration

                    sample_end = utt_end
                    
                    # Add to sample list
                    self.samples.append(current_sample)
                    self.sample_info.append(
                        {
                        "session_ID": session,
                        "start": sample_start,
                        "end": sample_end,
                        "n_speakers": len(set([x[0] for x in current_sample]))
                        }
                    )

                    current_sample = None # Reset for new sample


    def __len__(self) -> int:
        """Computes number of samples in dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float, float]], dict]: 

        """Get sample from dataset by index
        
        Args:
            idx (int): index of desired samples

        Returns:
            sample (tuple): first element is the audio waveform as a torch.Tensor with shape = (channels, samples)
            Second item is a list of true diarization intervals of the form (speaker, start, end). 
            Third item is a dict with additional sample info
        """
        
        sample, sample_info = self.samples[idx], self.sample_info[idx]

        # TODO add multi-channel support
        # Load audio, assummes a single-channel beamformed file named '"{session_ID}_U01.beamform.wav"'
        audio_path = os.path.join(self.audio_dir, f"{sample_info['session_ID']}_U01.beamform.wav")
        
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

    root_path = '/proj/disney/Dipco'
    dset = DinnerParty(root_path=root_path)

    print('Dataset generation complete')

    audio, segments, _ = dset[ randint(0, len(dset)) ]
    
    for segment in segments:
        print("SPEAKER %s <%.2f - %.2f>" % segment)

    torchaudio.save('sample.wav', audio, 16000)



