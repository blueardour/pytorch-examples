
from .mobilenet import MobileNetV2, MobileNetV1

__all__ = ['models', 'model_names']

model_names = ['mobilenet-v2', 'mobilenet-v1']

def models(arch):
    if arch == "mobilenet-v2":
        return MobileNetV2()
    elif arch == "mobilenet-v1":
        return MobileNetV1()
    else:
        return None


