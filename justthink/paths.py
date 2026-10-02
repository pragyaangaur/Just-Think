import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIN_AXIS = ROOT / "external" / "Pain-axis"
PAIN_AXIS_COMMIT = "4d75cd90e206ea962f7a9101e65c85efea56723b"
MODELS = ROOT / "models"
# Kaggle writes results outside the code checkout, so the output folder holds only results.
RESULTS = Path(os.environ.get("JUST_THINK_RESULTS", ROOT / "results"))


def pain_vector_file(pain_axis_name):
    return PAIN_AXIS / "results/3.2_pain_vectors/pain_vectors" / pain_axis_name / "pain_vectors.pt"
