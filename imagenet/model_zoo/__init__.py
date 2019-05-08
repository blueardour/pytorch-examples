
from .mobilenet import MobileNetV2, MobileNetV1
from .AlexNet import AlexNet
from .AlexNetBN import AlexNetBN

__all__ = ['models', 'model_names']

model_names = ['mobilenet-v2', 'mobilenet-v1', 'alexnet', 'alexnet_bn']

def models(arch):
    if arch == "mobilenet-v2":
        return MobileNetV2()
    elif arch == "mobilenet-v1":
        return MobileNetV1()
    elif arch == "alexnet":
        return AlexNet()
    elif arch == "alexnet_bn":
        return AlexNetBN()
    else:
        return None


