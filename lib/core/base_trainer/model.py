from train_config import config as cfg

from lib.core.base_trainer.models import PerchContinuousSED


MODEL_REGISTRY = {
    'perch_continuous_sed': PerchContinuousSED,
}


def get_model_class(arch=None):
    arch = arch or cfg.MODEL.arch
    if arch not in MODEL_REGISTRY:
        raise KeyError(f'Unknown model arch: {arch}. Available: {list(MODEL_REGISTRY.keys())}')
    return MODEL_REGISTRY[arch]


Net = get_model_class()


if __name__ == '__main__':
    import torch
    model = Net(weights_path=None)
    model.eval()
    dummy_audio = torch.randn(2, 32000 * 5)
    with torch.no_grad():
        logits, _ = model(dummy_audio)
    print(f'input:  {dummy_audio.shape}')
    print(f'output: {logits.shape}')
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
