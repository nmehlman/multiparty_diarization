from multiparty_diarization.utils import load_configs
from multiparty_diarization.models.pyannote_model import PyannoteDiarization
from multiparty_diarization.models.nemo_model import NEMO_Diarization
from multiparty_diarization.datasets.ami import AMI
from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics

import tqdm
import json
import pdb

import os

DATASETS = {
    "ami": AMI
}

MODELS = {
    "pyannote": PyannoteDiarization,
    "nemo": NEMO_Diarization
}

CONFIG_PATH = "./configs/ami_nemo.yaml"

if __name__ == "__main__":

    # Load eval config
    config = load_configs(CONFIG_PATH)

    # Load model and dataset
    model = MODELS[ config['model'] ](**config['model_kwargs'])
    dataset = DATASETS[ config['dataset'] ](**config["dataset_kwargs"])

    sample_results = []
    for audio, reference, info in tqdm.tqdm(dataset, total=len(dataset), desc='running diarization'):
        
        # Run diarization and compute metrics
        try:
            hypothesis = model(audio, **config['diarize_kwargs'])
            results = compute_sample_diarization_metrics(reference, hypothesis)
        except IndexError: # Handel corner cases 
            results = {
                "der": 1.0,
                "miss": 1.0,
                "false_alarm": 0.0,
                "confidence": 0.0,
            }
            
        results['info'] = info

        sample_results.append(results)

    # Save
    json.dump(
        sample_results,
        open( os.path.join(config['save_path'], 'raw_results.json'), 'w'),
        indent = 3
    )

    json.dump(
        sample_results,
        open( os.path.join(config['save_path'], 'config.json'), 'w'),
        indent = 3
    )



