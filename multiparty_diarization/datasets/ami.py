from dataset_wrapper import DiarizationDataset
from xml.etree import ElementTree
import librosa
import xml.etree.ElementTree as ET
import torch

from typing import List, Tuple
import os

class AMI(DiarizationDataset):

    def __init__(self, 
                root_path: str, 
                sample_len_s: float = 15.0,
                min_speaker_gap_s: float = 1.0 
                ):

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
        self._generate_sample(meeting_word_XML_files)

    def _generate_sample(self, meeting_word_XML_files: dict):

        self.samples = []
        self.sample_info = []

        for meeting_ID, meeting_files in meeting_word_XML_files.items(): # For each meeting
            
            speaker_segments = {}

            for speaker_word_file in meeting_files: # For each speaker
                speakerID = speaker_word_file.split('.')[-3]
                speaker_segments[speakerID] = self._parse_speaker_segments(speaker_word_file)

            print(speaker_segments)

    def _parse_speaker_segments(self, speaker_word_file: str):

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

            




    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float, float]]]: 
        
        sample, sample_info = self.samples[idx], self.sample_info[idx]

        # TODO add multi-channel support
        # Load audio, assummes a single-channel beamformed file named 'beamform.wav'
        audio_path = os.path.join(self.root_path, sample_info['meeting_ID'], 'audio', "beamform.wav")
        
        audio, _ = librosa.load(audio_path, 
                offset = sample_info['starttime'], 
                duration = sample_info['endtime'] - sample_info['starttime'],
                sr = 16000
                )

        audio = torch.from_numpy(audio).unsqueeze(0)

        return audio, sample
                
if __name__ == "__main__":

    from random import randint
    import torchaudio

    root_path = '/proj/disney/amicorpus'
    dset = AMI(root_path=root_path)

    #audio, segments = dset[ randint(0, len(dset)) ]
    
    #for segment in segments:
        #print("SPEAKER %s <%.2f - %.2f>" % segment)

    #torchaudio.save('sample.wav', audio, 16000)



