"""Architecture used in training (Week 2): smp model with a 12-channel first conv + input standardisation."""
import torch.nn as nn
import segmentation_models_pytorch as smp

ARCHS = {"Unet": smp.Unet, "DeepLabV3Plus": smp.DeepLabV3Plus}


class SMPWrap(nn.Module):
    """Inputs arrive min-max normalised to [0, 1]; standardise, then run the smp model."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m((x - 0.5) / 0.25)


def build_model(arch="DeepLabV3Plus", encoder="resnet34", in_channels=12):
    # encoder_weights=None: pretrained ImageNet weights are NOT needed at inference,
    # the fine-tuned weights come from the checkpoint. Keys match the training model.
    m = ARCHS[arch](encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    first = m.encoder.conv1
    m.encoder.conv1 = nn.Conv2d(in_channels, first.out_channels, first.kernel_size,
                                first.stride, first.padding, bias=first.bias is not None)
    return SMPWrap(m)
