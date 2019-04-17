
from .mobilenet import MobileNetV2

__all__ = ['models', 'model_names']

model_names = ['mobilenet-v2']

def models(arch):
    if arch == "mobilenet-v2":
        return MobileNetV2()
    else:
        return None


