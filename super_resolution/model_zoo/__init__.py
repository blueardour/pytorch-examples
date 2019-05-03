
from .vdsr import VDSR

__all__ = ['models', 'model_names']

model_names = ['VDSR']

def models(arch):
    if arch == "VDSR":
        return VDSR()
    else:
        return None


