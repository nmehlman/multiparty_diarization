from multiparty_diarization.utils import load_configs
from multiparty_diarization.models.single_channel_models.pyannote_model import PyannoteDiarization
from multiparty_diarization.models.single_channel_models.nemo_model import NEMO_Diarization
from multiparty_diarization.models.multi_channel_models.late_fusion_VAD.late_fusion_VAD_model import LateFusionVAD
from multiparty_diarization.models.multi_channel_models.em_vad.expectation_maximization_vad import EMCloseTalkVAD
from multiparty_diarization.datasets.ami import AMI
from multiparty_diarization.datasets.ami_multichannel import AMIMultiChannel
from multiparty_diarization.datasets.dinner_party import DinnerParty
from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics

import tqdm
import json
import sys

import os

DATASETS = {
    "ami": AMI,
    "dipco": DinnerParty,
    "ami_multichannel": AMIMultiChannel
}

MODELS = {
    "pyannote": PyannoteDiarization,
    "nemo": NEMO_Diarization,
    "vad_fusion": LateFusionVAD,
    "em_vad": EMCloseTalkVAD
}

if __name__ == "__main__":

    CONFIG_PATH = sys.argv[1]

    # Load eval config
    config = load_configs(CONFIG_PATH)

    # Load model and dataset
    model = MODELS[ config['model'] ](**config['model_kwargs'])
    dataset = DATASETS[ config['dataset'] ](**config["dataset_kwargs"])
    diarize_kwargs = config.get('diarize_kwargs', {})

    sample_results = []
    for i, (audio, reference, info) in tqdm.tqdm(
                                    enumerate(dataset), 
                                    total=len(dataset), 
                                    desc='running diarization'
                                ):
        
        # Run diarization and compute metrics
        try:
            if config['use_oracle']: # Use oracle info
                oracle_info = dataset.generate_oracle_info(i)
                hypothesis = model(audio, **diarize_kwargs, **oracle_info)
            
            else: # No oracle info
                hypothesis = model(audio, **diarize_kwargs)
            
            results = compute_sample_diarization_metrics(reference, hypothesis)
            
        except IndexError: # Handel corner cases 
            results = {
                "der": 1.0,
                "miss": 1.0,
                "false_alarm": 1.0,
                "confusion": 1.0,
            }
            
        results['info'] = info

        sample_results.append(results)

    # Save
    json.dump(
        sample_results,
        open( os.path.join(config['save_dir'], 'raw_results.json'), 'w'),
        indent = 3
    )

    json.dump(
        sample_results,
        open( os.path.join(config['save_dir'], 'config.json'), 'w'),
        indent = 3
    )



