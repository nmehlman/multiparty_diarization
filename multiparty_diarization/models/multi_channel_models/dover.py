import numpy as np 
from typing import Tuple, List
from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics
from itertools import permutations

def pad_labels(labs_1: set, labs_2: set):

    if len(labs_1) > len(labs_2):
        for i in range(len(labs_1) - len(labs_2)):
            labs_2.add(f'unmapped_{i+1}')

    elif len(labs_1) < len(labs_2):
        for i in range(len(labs_2) - len(labs_1)):
            labs_1.add(f'unmapped_{i+1}')

    assert len(labs_1) == len(labs_2)

    return labs_1, labs_2
 
def compute_best_label_mapping(
        output_1: List[ Tuple[str, float, float] ], 
        output_2: List[ Tuple[str, float, float] ]
        ) -> dict:
    
    # Get unique speaker labels for each output
    speaker_labs_1 = set([x[0] for x in output_1])
    speaker_labs_2 = set([x[0] for x in output_2])

    speaker_labs_1, speaker_labs_2 = pad_labels(speaker_labs_1, speaker_labs_2) # Match number of speakers

    for new_labs_2 in permutations(speaker_labs_2):

        lab_mapping = dict(zip(speaker_labs_2, new_labs_2))
        


def dover(diarization_outputs: Tuple[ List[ Tuple[str, float, float] ] ] , **kwargs):

    N = len(diarization_outputs) # Number of diarization outputs

    # Get system weights if present
    weights = kwargs.get('weights', N * [1])
    assert len(weights) == N, 'must have one weight per diarization output'
    assert np.min(weights) >= 0, 'weights must be non-negative'

    # Label mapping
    for i in range(1, N):
        for j in range(i):
