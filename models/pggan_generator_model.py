# python3.7
"""Contains the implementation of generator described in ProgressiveGAN.

Different from the official tensorflow model in folder `pggan_tf_official`, this
is a simple pytorch version which only contains the generator part. This class
is specially used for inference.

For more details, please check the original paper:
https://arxiv.org/pdf/1710.10196.pdf
"""

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

__all__ = ['PGGANGeneratorModel']

# Defines a dictionary, which maps the target resolution of the final generated
# image to numbers of filters used in each convolutional layer in sequence.
_RESOLUTIONS_TO_CHANNELS = {
    8: [512, 512, 512],
    16: [512, 512, 512, 512],
    32: [512, 512, 512, 512, 512],
    64: [512, 512, 512, 512, 512, 256],
    128: [512, 512, 512, 512, 512, 256, 128],
    256: [512, 512, 512, 512, 512, 256, 128, 64],
    512: [512, 512, 512, 512, 512, 256, 128, 64, 32],
    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16],
}

# Variable mapping from pytorch model to official tensorflow model.
_PGGAN_PTH_VARS_TO_TF_VARS = {
    'layer1.conv.weight': '4x4/Dense/weight',  # [512, 512, 4, 4]
    'layer1.wscale.bias': '4x4/Dense/bias',  # [512]
    'layer2.conv.weight': '4x4/Conv/weight',  # [512, 512, 3, 3]
    'layer2.wscale.bias': '4x4/Conv/bias',  # [512]
    'layer3.conv.weight': '8x8/Conv0/weight',  # [512, 512, 3, 3]
    'layer3.wscale.bias': '8x8/Conv0/bias',  # [512]
    'layer4.conv.weight': '8x8/Conv1/weight',  # [512, 512, 3, 3]
    'layer4.wscale.bias': '8x8/Conv1/bias',  # [512]
    'layer5.conv.weight': '16x16/Conv0/weight',  # [512, 512, 3, 3]
    'layer5.wscale.bias': '16x16/Conv0/bias',  # [512]
    'layer6.conv.weight': '16x16/Conv1/weight',  # [512, 512, 3, 3]
    'layer6.wscale.bias': '16x16/Conv1/bias',  # [512]
    'layer7.conv.weight': '32x32/Conv0/weight',  # [512, 512, 3, 3]
    'layer7.wscale.bias': '32x32/Conv0/bias',  # [512]
    'layer8.conv.weight': '32x32/Conv1/weight',  # [512, 512, 3, 3]
    'layer8.wscale.bias': '32x32/Conv1/bias',  # [512]
    'layer9.conv.weight': '64x64/Conv0/weight',  # [256, 512, 3, 3]
    'layer9.wscale.bias': '64x64/Conv0/bias',  # [256]
    'layer10.conv.weight': '64x64/Conv1/weight',  # [256, 256, 3, 3]
    'layer10.wscale.bias': '64x64/Conv1/bias',  # [256]
    'layer11.conv.weight': '128x128/Conv0/weight',  # [128, 256, 3, 3]
    'layer11.wscale.bias': '128x128/Conv0/bias',  # [128]
    'layer12.conv.weight': '128x128/Conv1/weight',  # [128, 128, 3, 3]
    'layer12.wscale.bias': '128x128/Conv1/bias',  # [128]
    'layer13.conv.weight': '256x256/Conv0/weight',  # [64, 128, 3, 3]
    'layer13.wscale.bias': '256x256/Conv0/bias',  # [64]
    'layer14.conv.weight': '256x256/Conv1/weight',  # [64, 64, 3, 3]
    'layer14.wscale.bias': '256x256/Conv1/bias',  # [64]
    'layer15.conv.weight': '512x512/Conv0/weight',  # [32, 64, 3, 3]
    'layer15.wscale.bias': '512x512/Conv0/bias',  # [32]
    'layer16.conv.weight': '512x512/Conv1/weight',  # [32, 32, 3, 3]
    'layer16.wscale.bias': '512x512/Conv1/bias',  # [32]
    'layer17.conv.weight': '1024x1024/Conv0/weight',  # [16, 32, 3, 3]
    'layer17.wscale.bias': '1024x1024/Conv0/bias',  # [16]
    'layer18.conv.weight': '1024x1024/Conv1/weight',  # [16, 16, 3, 3]
    'layer18.wscale.bias': '1024x1024/Conv1/bias',  # [16]
    'output_1024x1024.conv.weight': 'ToRGB_lod0/weight',  # [3, 16, 1, 1]
    'output_1024x1024.wscale.bias': 'ToRGB_lod0/bias',  # [3]
}


class PGGANGeneratorModel(nn.Sequential):
  """Defines the generator module in ProgressiveGAN.

  Note that the generated images are with RGB color channels with range [-1, 1].
  """

  def __init__(self, resolution=1024, final_tanh=False):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the final output image. (default: 1024)
      final_tanh: Whether to use a `tanh` function to clamp the pixel values of
        the output image to range [-1, 1]. (default: False)

    Raises:
      ValueError: If the input `resolution` is not supported.
    """
    try:
      channels = _RESOLUTIONS_TO_CHANNELS[resolution]
    except KeyError:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: '
                       f'{list(_RESOLUTIONS_TO_CHANNELS)}.')

    sequence = OrderedDict()

    def _add_layer(layer, name=None):
      name = name or f'layer{len(sequence) + 1}'
      sequence[name] = layer

    _add_layer(ConvBlock(channels[0], channels[1], kernel_size=4, padding=3))
    _add_layer(ConvBlock(channels[1], channels[1]))
    for i in range(2, len(channels)):
      _add_layer(ConvBlock(channels[i-1], channels[i], upsample=True))
      _add_layer(ConvBlock(channels[i], channels[i]))
    # Final convolutional block.
    _add_layer(ConvBlock(in_channels=channels[-1],
                         out_channels=3,
                         kernel_size=1,
                         padding=0,
                         wscale_gain=1.0,
                         activation_type='tanh' if final_tanh else 'linear'),
               name=f'output_{resolution}x{resolution}')
    super().__init__(sequence)
    self.pth_to_tf_var_mapping = _PGGAN_PTH_VARS_TO_TF_VARS

  def forward(self, x):
    if len(x.shape) != 2:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'noise_dim], but {x.shape} received!')
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    return super().forward(x)


class PixelNormLayer(nn.Module):
  """Implements pixel-wise feature vector normalization layer."""

  def __init__(self, epsilon=1e-8):
    super().__init__()
    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ResolutionScalingLayer(nn.Module):
  """Implements the resolution scaling layer.

  Basically, this layer can be used to upsample or downsample feature maps from
  spatial domain with nearest neighbor interpolation.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    return nn.functional.interpolate(x,
                                     scale_factor=self.scale_factor,
                                     mode='nearest')


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  Note that, the weight variable is trained in `nn.Conv2d` layer, and only
  scaled with a constant number, which is not trainable, in this layer. However,
  the bias variable is trainable in this layer.
  """

  def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in)
    self.bias = nn.Parameter(torch.zeros(out_channels))

  def forward(self, x):
    return x * self.scale + self.bias.view(1, -1, 1, 1)


class ConvBlock(nn.Module):
  """Implements the convolutional block used in ProgressiveGAN.

  Basically, this block executes pixel-wise normalization layer, upsampling
  layer (if needed), convolutional layer, weight-scale layer, and activation
  layer in sequence.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               upsample=False,
               wscale_gain=np.sqrt(2.0),
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      upsample: Whether to upsample the input tensor before convolution.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      activation_type: Type of activation function. Support `linear`, `lrelu`
        and `tanh`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()
    self.pixel_norm = PixelNormLayer()
    self.upsample = ResolutionScalingLayer() if upsample else (lambda x: x)
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=1,
                          bias=add_bias)
    self.wscale = WScaleLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              gain=wscale_gain)
    if activation_type == 'linear':
      self.activate = (lambda x: x)
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation_type == 'tanh':
      self.activate = nn.Hardtanh()
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    x = self.pixel_norm(x)
    x = self.upsample(x)
    x = self.conv(x)
    x = self.wscale(x)
    x = self.activate(x)
    return x

