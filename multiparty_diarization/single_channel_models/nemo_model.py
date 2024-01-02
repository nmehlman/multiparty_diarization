from typing import List, Tuple
import torch
from multiparty_diarization.single_channel_models.model_wrapper import DiarizationModel
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
import yaml
from multiparty_diarization.datasets.dataset_wrapper import DiarizationDataset
from omegaconf import DictConfig
import os
import json
from shutil import rmtree

import torchaudio

class NEMO_Diarization(DiarizationModel):

    def __init__(self,
                config_path: str,
                pretrained_vad_model: str = 'vad_multilingual_marblenet',
                pretrained_speaker_model: str = 'titanet_large',
                pretrained_neural_diarizer_model: str = 'diar_msdd_telephonic',
                use_oracle_vad: bool = False,
                use_oracle_num_speakers: bool = False,
                tmp_dir: str = "./tmp",
                device: str = 'cpu'  
            ):
        
        super().__init__()

        self.tmp_dir = tmp_dir
        self.device = device
        self.use_oracle_vad = use_oracle_vad
        self.use_oracle_num_speakers = use_oracle_num_speakers

        # Load config
        config = DictConfig(
            yaml.load(
                open(config_path), 
                Loader=yaml.SafeLoader
                )
            )
        
        # Modify config based on kwargs
        config.diarizer.vad.model_path=pretrained_vad_model
        config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model 
        config.diarizer.msdd_model.model_path=pretrained_neural_diarizer_model
        config.diarizer.clustering.parameters.oracle_num_speakers=use_oracle_num_speakers
        config.diarizer.oracle_vad=use_oracle_vad

        # Set temp directory for model outputs
        config.diarizer.out_dir = self.tmp_dir

        # Load model
        self._model = ClusteringDiarizer(config).to(device)

    def __call__(self, waveform: torch.tensor, sample_rate: int, **oracle_info) -> List[Tuple[str, float, float]]:
        
        # Make tmp directory if needed
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        # Need to save waveform to file to conform with NEMO interface
        audio_path = os.path.join(self.tmp_dir, 'tmp_audio.wav')
        torchaudio.save( audio_path, waveform, sample_rate ) 

        if self.use_oracle_vad or self.use_oracle_num_speakers: # Generate manifest

            assert oracle_info, 'oracle info required with use_oracle_vad set to True'

            # Create RTTM file
            rttm_path = os.path.join(self.tmp_dir, 'segment.rttm')
            with open(rttm_path, 'w') as rttm_file:
                for segment in oracle_info['segments']:
                    speaker, start, end = segment
                    line = f"SPEAKER {audio_path} 1 {start} {end} <NA> <NA> {speaker} <NA> <NA>\n"
                    rttm_file.write(line)

            # Create manifest
            manifest_path = os.path.join(self.tmp_dir, 'manifest.json')
            manifest = {
                'audio_filepath': audio_path, 
                'offset': 0, 
                'duration': None, 
                'label': 'infer', 
                'text': '-', 
                'num_speakers': oracle_info['n_speakers'], 
                'rttm_filepath': rttm_path, 
                'uem_filepath' : None
            }

            json.dump(manifest, open(manifest_path, 'w'))

            self._model._diarizer_params.manifest_filepath = manifest_path # Link manifest

            self._model.diarize() # Run diarization

        else: # No oracle info, run on raw audio
            audio_files = [audio_path] # Run diarization
            self._model.diarize(audio_files)

        labels = rttm_to_labels(os.path.join(self.tmp_dir, 'pred_rttms', 'tmp_audio.rttm'))
        diarization = labels_to_pyannote_object(labels)

        results = [ (spkr.split('_')[-1], speech_turn.start, speech_turn.end)
                for speech_turn, _, spkr in diarization.itertracks(yield_label=True)
                ]
        
        # Remove tmp directory
        rmtree(self.tmp_dir)

        return results


if __name__ == "__main__":

    import os
    import wget
    import torchaudio
    from shutil import rmtree
    from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics
    
    ROOT = "/home/nmehlman/disney/multiparty_diarization/misc"
    data_dir = os.path.join(ROOT,'data')
    os.makedirs(data_dir, exist_ok=True)
    an4_audio = os.path.join(data_dir,'an4_diarize_test.wav')
    an4_rttm = os.path.join(data_dir,'an4_diarize_test.rttm')
    
    if not os.path.exists(an4_audio):
        an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
        an4_audio = wget.download(an4_audio_url, data_dir)
    if not os.path.exists(an4_rttm):
        an4_rttm_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
        an4_rttm = wget.download(an4_rttm_url, data_dir)
    
    waveform, fs = torchaudio.load('../../misc/data/an4_diarize_test.wav')

    labels = rttm_to_labels(an4_rttm) # True labels
    reference = [ (spkr.split('_')[-1], speech_turn.start, speech_turn.end)
                for speech_turn, _, spkr 
                in labels_to_pyannote_object(labels).itertracks(yield_label=True)
                ]
    
    model = NEMO_Diarization(config_path='./nemo_config.yaml', device='cuda:0')
    
    hypothesis = model(waveform, fs)
    
    metrics = compute_sample_diarization_metrics(reference, hypothesis)

    print(metrics)

    rmtree('./tmp')