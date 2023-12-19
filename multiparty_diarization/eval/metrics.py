import spyder
from typing import List, Tuple

# See https://github.com/desh2608/spyder

def compute_sample_diarization_metrics(
        reference: List[Tuple[str, float, float]], 
        hypothesis: List[Tuple[str, float, float]]
        ) -> dict:
    
    metrics = spyder.DER(ref=reference, hyp=hypothesis)

    results = {
        "der": metrics.der,
        "miss": metrics.miss,
        "false_alarm": metrics.falarm,
        "confusion": metrics.conf
    }

    return results

