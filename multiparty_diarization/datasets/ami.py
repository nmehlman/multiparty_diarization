from multiparty_diarization.datasets.dataset_wrapper import DiarizationDataset
from xml.etree import ElementTree
import librosa
import xml.etree.ElementTree as ET
import torch

from typing import List, Tuple, Dict
import os

class AMI(DiarizationDataset):

    """Class for AMI Dataset"""

    def __init__(self, 
                root_path: str, 
                sample_len_s: float = 15.0,
                min_speaker_gap_s: float = 1.0 
                ):
        
        """Create instance of AMI dataset
        
        Args:
            root_path (str): path to AMI dataset directory
            sample_len_s (float): target length of each sample
            min_speaker_gap_s: minimum interval of non-speech to separate speaker segments
        """

        self.root_path = root_path
        self.sample_len_s = sample_len_s
        self.min_speaker_gap_s = min_speaker_gap_s

        word_dir = os.path.join(root_path, 'words') # Contains transcript info
        meeting_IDs = [subdir for subdir in next(os.walk(root_path))[1] if subdir[0].isupper()] # List all meetings

        meeting_word_XML_files = { # List XML transcript files associated each meeting (one per participant)
            meeting_ID: [
                    os.path.join(word_dir, fname) for fname in os.listdir(word_dir) if fname.split('.')[0] == meeting_ID
                    ]
                for meeting_ID in meeting_IDs
            }

        # Generate samples
        self._generate_samples(meeting_word_XML_files)

    def _generate_samples(self, meeting_word_XML_files: Dict[str, List]):
        
        """Generates diarization samples
        
        Args:
            meeting_word_XML_files (dict): dict mapping meeting IDs to list of XML files (one per speaker)
            that contains the word-level segmentations.
        """

        self.samples = []
        self.sample_info = []

        for meeting_ID, meeting_files in meeting_word_XML_files.items(): # For each meeting
            
            speaker_segments = {} # Dict of segments for each speaker in meeting

            for speaker_word_file in meeting_files: # For each speaker
                speakerID = speaker_word_file.split('.')[-3]
                speaker_segments[speakerID] = self._parse_speaker_segments(speaker_word_file)

            meeting_samples, meeting_sample_info = self._generate_meeting_samples(speaker_segments, meeting_ID)

            self.samples += meeting_samples
            self.sample_info += meeting_sample_info

    def _parse_speaker_segments(self, speaker_word_file: str) -> List[Tuple[float, float]]:

        """Determine speech intervals for a given speaker
        
        Args:
            speaker_word_file (str): path to word-level XML file for speaker

        Returns:
            speaker_segments (list): list of tuples marking speech intervals [(start, end)]
        """

        tree = ET.parse(speaker_word_file)
        root = tree.getroot()

        segments = []
        current_seg_start = None
        prev_word_end = None

        for word in root:

            if word.tag != 'w' or 'starttime' not in word.attrib:
                continue

            if current_seg_start is None:
                current_seg_start = float(word.attrib['starttime'])
                
            if prev_word_end is not None:
                inter_word_gap = float(word.attrib['starttime']) - prev_word_end

                if inter_word_gap > self.min_speaker_gap_s: # Start new segment
                    segments.append((current_seg_start, prev_word_end))

                    current_seg_start = float(word.attrib['starttime'])

            prev_word_end = float(word.attrib['endtime'])

        return segments
    
    def _generate_meeting_samples(self, speaker_segments: dict, meeting_ID: str) -> Tuple[list, list]:

        """Generates samples for a single meeting
        
        Args:
            speaker_segments (dict): dictionary mapping speaker ID to their speaking segments
            
            meeting_ID (str): meeting ID

        Returns:
            (list, list): list of samples (speaker, start, end) and list of info dict for each sample
        """

        samples = []
        sample_info = []
        
        # Merge speaker segments
        merged_speaker_segments = []
        for id, segs in speaker_segments.items():
            merged_speaker_segments += [(id, start, end) for start, end in segs]
        
        # Sort segments by start time
        merged_speaker_segments = sorted(merged_speaker_segments, key = lambda seg: seg[1])

        n_segments = len(merged_speaker_segments)
        sample_assigments = [None] * n_segments
        current_segment_idx = 0
        sample_start = None
        sample_end = None

        for i in range(n_segments):
            
            if sample_assigments[i] is None: # Sample not yet assigned
                sample_assigments[i] = current_segment_idx
                
                if sample_start is None:
                    sample_start = merged_speaker_segments[i][1]

                sample_end = merged_speaker_segments[i][-1]

                # Check length
                if sample_end - sample_start > self.sample_len_s: 

                    for j in range(i+1, n_segments):
                        if (sample_assigments[j] is None) and (merged_speaker_segments[j][-1] < sample_end):
                            sample_assigments[j] = current_segment_idx 

                    current_segment_idx += 1 # Start new sample
                    sample_start = None
            
        for k in range(max(sample_assigments)):

            sample = [seg for samp_assign, seg in zip(sample_assigments, merged_speaker_segments) if samp_assign == k]
            self.samples.append(sample)
            self.sample_info.append(
                {
                    "meeting_ID": meeting_ID,
                    "start": min([x[1] for x in sample]),
                    "end": max([x[2] for x in sample]),
                    "n_speakers": len(set([x[0] for x in sample]))
                }
            )

        return samples, sample_info
    
    def _normalize_sample(self, sample, sample_info):
        start = sample_info['start']
        sample = [(spkr, raw_start - start, raw_end - start) for spkr, raw_start, raw_end in sample]
        return sample

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

    root_path = '/proj/disney/amicorpus'
    dset = AMI(root_path=root_path)

    print('Dataset generation complete')

    audio, segments = dset[ randint(0, len(dset)) ]
    
    for segment in segments:
        print("SPEAKER %s <%.2f - %.2f>" % segment)

    torchaudio.save('sample.wav', audio, 16000)



