# python3.7
"""Contains the implementation of generator described in ProgressiveGAN.

Different from the official tensorflow model in folder `pggan_tf_official`, this
is a simple pytorch version which only contains the generator part. This class
is specially used for inference.

For more details, please check the original paper:
https://arxiv.org/pdf/1710.10196.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    'lod': 'lod',  # []
    'layer0.conv.weight': '4x4/Dense/weight',  # [512, 512, 4, 4]
    'layer0.wscale.bias': '4x4/Dense/bias',  # [512]
    'layer1.conv.weight': '4x4/Conv/weight',  # [512, 512, 3, 3]
    'layer1.wscale.bias': '4x4/Conv/bias',  # [512]
    'layer2.conv.weight': '8x8/Conv0/weight',  # [512, 512, 3, 3]
    'layer2.wscale.bias': '8x8/Conv0/bias',  # [512]
    'layer3.conv.weight': '8x8/Conv1/weight',  # [512, 512, 3, 3]
    'layer3.wscale.bias': '8x8/Conv1/bias',  # [512]
    'layer4.conv.weight': '16x16/Conv0/weight',  # [512, 512, 3, 3]
    'layer4.wscale.bias': '16x16/Conv0/bias',  # [512]
    'layer5.conv.weight': '16x16/Conv1/weight',  # [512, 512, 3, 3]
    'layer5.wscale.bias': '16x16/Conv1/bias',  # [512]
    'layer6.conv.weight': '32x32/Conv0/weight',  # [512, 512, 3, 3]
    'layer6.wscale.bias': '32x32/Conv0/bias',  # [512]
    'layer7.conv.weight': '32x32/Conv1/weight',  # [512, 512, 3, 3]
    'layer7.wscale.bias': '32x32/Conv1/bias',  # [512]
    'layer8.conv.weight': '64x64/Conv0/weight',  # [256, 512, 3, 3]
    'layer8.wscale.bias': '64x64/Conv0/bias',  # [256]
    'layer9.conv.weight': '64x64/Conv1/weight',  # [256, 256, 3, 3]
    'layer9.wscale.bias': '64x64/Conv1/bias',  # [256]
    'layer10.conv.weight': '128x128/Conv0/weight',  # [128, 256, 3, 3]
    'layer10.wscale.bias': '128x128/Conv0/bias',  # [128]
    'layer11.conv.weight': '128x128/Conv1/weight',  # [128, 128, 3, 3]
    'layer11.wscale.bias': '128x128/Conv1/bias',  # [128]
    'layer12.conv.weight': '256x256/Conv0/weight',  # [64, 128, 3, 3]
    'layer12.wscale.bias': '256x256/Conv0/bias',  # [64]
    'layer13.conv.weight': '256x256/Conv1/weight',  # [64, 64, 3, 3]
    'layer13.wscale.bias': '256x256/Conv1/bias',  # [64]
    'layer14.conv.weight': '512x512/Conv0/weight',  # [32, 64, 3, 3]
    'layer14.wscale.bias': '512x512/Conv0/bias',  # [32]
    'layer15.conv.weight': '512x512/Conv1/weight',  # [32, 32, 3, 3]
    'layer15.wscale.bias': '512x512/Conv1/bias',  # [32]
    'layer16.conv.weight': '1024x1024/Conv0/weight',  # [16, 32, 3, 3]
    'layer16.wscale.bias': '1024x1024/Conv0/bias',  # [16]
    'layer17.conv.weight': '1024x1024/Conv1/weight',  # [16, 16, 3, 3]
    'layer17.wscale.bias': '1024x1024/Conv1/bias',  # [16]
    'output0.conv.weight': 'ToRGB_lod8/weight',  # [3, 512, 1, 1]
    'output0.wscale.bias': 'ToRGB_lod8/bias',  # [3]
    'output1.conv.weight': 'ToRGB_lod7/weight',  # [3, 512, 1, 1]
    'output1.wscale.bias': 'ToRGB_lod7/bias',  # [3]
    'output2.conv.weight': 'ToRGB_lod6/weight',  # [3, 512, 1, 1]
    'output2.wscale.bias': 'ToRGB_lod6/bias',  # [3]
    'output3.conv.weight': 'ToRGB_lod5/weight',  # [3, 512, 1, 1]
    'output3.wscale.bias': 'ToRGB_lod5/bias',  # [3]
    'output4.conv.weight': 'ToRGB_lod4/weight',  # [3, 256, 1, 1]
    'output4.wscale.bias': 'ToRGB_lod4/bias',  # [3]
    'output5.conv.weight': 'ToRGB_lod3/weight',  # [3, 128, 1, 1]
    'output5.wscale.bias': 'ToRGB_lod3/bias',  # [3]
    'output6.conv.weight': 'ToRGB_lod2/weight',  # [3, 64, 1, 1]
    'output6.wscale.bias': 'ToRGB_lod2/bias',  # [3]
    'output7.conv.weight': 'ToRGB_lod1/weight',  # [3, 32, 1, 1]
    'output7.wscale.bias': 'ToRGB_lod1/bias',  # [3]
    'output8.conv.weight': 'ToRGB_lod0/weight',  # [3, 16, 1, 1]
    'output8.wscale.bias': 'ToRGB_lod0/bias',  # [3]
}


class PGGANGeneratorModel(nn.Module):
  """Defines the generator module in ProgressiveGAN.

  Note that the generated images are with RGB color channels with range [-1, 1].
  """

  def __init__(self,
               resolution=1024,
               fused_scale=False,
               output_channels=3):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the final output image. (default: 1024)
      fused_scale: Whether to fused `upsample` and `conv2d` together, resulting
        in `conv2_transpose`. (default: False)
      output_channels: Number of channels of the output image. (default: 3)

    Raises:
      ValueError: If the input `resolution` is not supported.
    """
    super().__init__()

    try:
      self.channels = _RESOLUTIONS_TO_CHANNELS[resolution]
    except KeyError:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: '
                       f'{list(_RESOLUTIONS_TO_CHANNELS)}.')
    assert len(self.channels) == int(np.log2(resolution))

    self.resolution = resolution
    self.fused_scale = fused_scale
    self.output_channels = output_channels

    for block_idx in range(1, len(self.channels)):
      if block_idx == 1:
        self.add_module(
            f'layer{2 * block_idx - 2}',
            ConvBlock(in_channels=self.channels[block_idx - 1],
                      out_channels=self.channels[block_idx],
                      kernel_size=4,
                      padding=3))
      else:
        self.add_module(
            f'layer{2 * block_idx - 2}',
            ConvBlock(in_channels=self.channels[block_idx - 1],
                      out_channels=self.channels[block_idx],
                      upsample=True,
                      fused_scale=self.fused_scale))
      self.add_module(
          f'layer{2 * block_idx - 1}',
          ConvBlock(in_channels=self.channels[block_idx],
                    out_channels=self.channels[block_idx]))
      self.add_module(
          f'output{block_idx - 1}',
          ConvBlock(in_channels=self.channels[block_idx],
                    out_channels=self.output_channels,
                    kernel_size=1,
                    padding=0,
                    wscale_gain=1.0,
                    activation_type='linear'))

    self.upsample = ResolutionScalingLayer()
    self.lod = nn.Parameter(torch.zeros(()))

    self.pth_to_tf_var_mapping = {}
    for pth_var_name, tf_var_name in _PGGAN_PTH_VARS_TO_TF_VARS.items():
      if self.fused_scale and 'Conv0' in tf_var_name:
        pth_var_name = pth_var_name.replace('conv.weight', 'weight')
        tf_var_name = tf_var_name.replace('Conv0', 'Conv0_up')
      self.pth_to_tf_var_mapping[pth_var_name] = tf_var_name

  def forward(self, x):
    if len(x.shape) != 2:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'noise_dim], but {x.shape} received!')
    x = x.view(x.shape[0], x.shape[1], 1, 1)

    lod = self.lod.cpu().tolist()
    for block_idx in range(1, len(self.channels)):
      if block_idx + lod < len(self.channels):
        x = self.__getattr__(f'layer{2 * block_idx - 2}')(x)
        x = self.__getattr__(f'layer{2 * block_idx - 1}')(x)
        image = self.__getattr__(f'output{block_idx - 1}')(x)
      else:
        image = self.upsample(image)
    return image


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
    return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


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
               fused_scale=False,
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
      fused_scale: Whether to fused `upsample` and `conv2d` together, resulting
        in `conv2_transpose`.
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

    if upsample and not fused_scale:
      self.upsample = ResolutionScalingLayer()
    else:
      self.upsample = nn.Identity()

    if upsample and fused_scale:
      self.weight = nn.Parameter(
          torch.randn(kernel_size, kernel_size, in_channels, out_channels))
      fan_in = in_channels * kernel_size * kernel_size
      self.scale = wscale_gain / np.sqrt(fan_in)
    else:
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
      self.activate = nn.Identity()
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
    if hasattr(self, 'conv'):
      x = self.conv(x)
    else:
      kernel = self.weight * self.scale
      kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
      kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                kernel[1:, :-1] + kernel[:-1, :-1])
      kernel = kernel.permute(2, 3, 0, 1)
      x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
      x = x / self.scale
    x = self.wscale(x)
    x = self.activate(x)
    return x
