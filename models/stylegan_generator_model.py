# python3.7
"""Contains the implementation of generator described in StyleGAN.

Different from the official tensorflow model in folder `stylegan_tf_official`,
this is a simple pytorch version which only contains the generator part. This
class is specially used for inference.

For more details, please check the original paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleGANGeneratorModel']

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

# pylint: disable=line-too-long
# Variable mapping from pytorch model to official tensorflow model.
_STYLEGAN_PTH_VARS_TO_TF_VARS = {
    # Statistic information of disentangled latent feature, w.
    'truncation.w_avg':'dlatent_avg',  # [512]

    # Noises.
    'synthesis.layer0.epilogue.apply_noise.noise': 'noise0',  # [1, 1, 4, 4]
    'synthesis.layer1.epilogue.apply_noise.noise': 'noise1',  # [1, 1, 4, 4]
    'synthesis.layer2.epilogue.apply_noise.noise': 'noise2',  # [1, 1, 8, 8]
    'synthesis.layer3.epilogue.apply_noise.noise': 'noise3',  # [1, 1, 8, 8]
    'synthesis.layer4.epilogue.apply_noise.noise': 'noise4',  # [1, 1, 16, 16]
    'synthesis.layer5.epilogue.apply_noise.noise': 'noise5',  # [1, 1, 16, 16]
    'synthesis.layer6.epilogue.apply_noise.noise': 'noise6',  # [1, 1, 32, 32]
    'synthesis.layer7.epilogue.apply_noise.noise': 'noise7',  # [1, 1, 32, 32]
    'synthesis.layer8.epilogue.apply_noise.noise': 'noise8',  # [1, 1, 64, 64]
    'synthesis.layer9.epilogue.apply_noise.noise': 'noise9',  # [1, 1, 64, 64]
    'synthesis.layer10.epilogue.apply_noise.noise': 'noise10',  # [1, 1, 128, 128]
    'synthesis.layer11.epilogue.apply_noise.noise': 'noise11',  # [1, 1, 128, 128]
    'synthesis.layer12.epilogue.apply_noise.noise': 'noise12',  # [1, 1, 256, 256]
    'synthesis.layer13.epilogue.apply_noise.noise': 'noise13',  # [1, 1, 256, 256]
    'synthesis.layer14.epilogue.apply_noise.noise': 'noise14',  # [1, 1, 512, 512]
    'synthesis.layer15.epilogue.apply_noise.noise': 'noise15',  # [1, 1, 512, 512]
    'synthesis.layer16.epilogue.apply_noise.noise': 'noise16',  # [1, 1, 1024, 1024]
    'synthesis.layer17.epilogue.apply_noise.noise': 'noise17',  # [1, 1, 1024, 1024]

    # Mapping blocks.
    'mapping.dense0.linear.weight': 'Dense0/weight',  # [512, 512]
    'mapping.dense0.wscale.bias': 'Dense0/bias',  # [512]
    'mapping.dense1.linear.weight': 'Dense1/weight',  # [512, 512]
    'mapping.dense1.wscale.bias': 'Dense1/bias',  # [512]
    'mapping.dense2.linear.weight': 'Dense2/weight',  # [512, 512]
    'mapping.dense2.wscale.bias': 'Dense2/bias',  # [512]
    'mapping.dense3.linear.weight': 'Dense3/weight',  # [512, 512]
    'mapping.dense3.wscale.bias': 'Dense3/bias',  # [512]
    'mapping.dense4.linear.weight': 'Dense4/weight',  # [512, 512]
    'mapping.dense4.wscale.bias': 'Dense4/bias',  # [512]
    'mapping.dense5.linear.weight': 'Dense5/weight',  # [512, 512]
    'mapping.dense5.wscale.bias': 'Dense5/bias',  # [512]
    'mapping.dense6.linear.weight': 'Dense6/weight',  # [512, 512]
    'mapping.dense6.wscale.bias': 'Dense6/bias',  # [512]
    'mapping.dense7.linear.weight': 'Dense7/weight',  # [512, 512]
    'mapping.dense7.wscale.bias': 'Dense7/bias',  # [512]

    # Synthesis blocks.
    'synthesis.lod': 'lod',  # []
    'synthesis.layer0.first_layer': '4x4/Const/const',  # [1, 512, 4, 4]
    'synthesis.layer0.epilogue.apply_noise.weight': '4x4/Const/Noise/weight',  # [512]
    'synthesis.layer0.epilogue.bias': '4x4/Const/bias',  # [512]
    'synthesis.layer0.epilogue.style_mod.dense.linear.weight': '4x4/Const/StyleMod/weight',  # [1024, 512]
    'synthesis.layer0.epilogue.style_mod.dense.wscale.bias': '4x4/Const/StyleMod/bias',  # [1024]
    'synthesis.layer1.conv.weight': '4x4/Conv/weight',  # [512, 512, 3, 3]
    'synthesis.layer1.epilogue.apply_noise.weight': '4x4/Conv/Noise/weight',  # [512]
    'synthesis.layer1.epilogue.bias': '4x4/Conv/bias',  # [512]
    'synthesis.layer1.epilogue.style_mod.dense.linear.weight': '4x4/Conv/StyleMod/weight',  # [1024, 512]
    'synthesis.layer1.epilogue.style_mod.dense.wscale.bias': '4x4/Conv/StyleMod/bias',  # [1024]
    'synthesis.layer2.conv.weight': '8x8/Conv0_up/weight',  # [512, 512, 3, 3]
    'synthesis.layer2.epilogue.apply_noise.weight': '8x8/Conv0_up/Noise/weight',  # [512]
    'synthesis.layer2.epilogue.bias': '8x8/Conv0_up/bias',  # [512]
    'synthesis.layer2.epilogue.style_mod.dense.linear.weight': '8x8/Conv0_up/StyleMod/weight',  # [1024, 512]
    'synthesis.layer2.epilogue.style_mod.dense.wscale.bias': '8x8/Conv0_up/StyleMod/bias',  # [1024]
    'synthesis.layer3.conv.weight': '8x8/Conv1/weight',  # [512, 512, 3, 3]
    'synthesis.layer3.epilogue.apply_noise.weight': '8x8/Conv1/Noise/weight',  # [512]
    'synthesis.layer3.epilogue.bias': '8x8/Conv1/bias',  # [512]
    'synthesis.layer3.epilogue.style_mod.dense.linear.weight': '8x8/Conv1/StyleMod/weight',  # [1024, 512]
    'synthesis.layer3.epilogue.style_mod.dense.wscale.bias': '8x8/Conv1/StyleMod/bias',  # [1024]
    'synthesis.layer4.conv.weight': '16x16/Conv0_up/weight',  # [512, 512, 3, 3]
    'synthesis.layer4.epilogue.apply_noise.weight': '16x16/Conv0_up/Noise/weight',  # [512]
    'synthesis.layer4.epilogue.bias': '16x16/Conv0_up/bias',  # [512]
    'synthesis.layer4.epilogue.style_mod.dense.linear.weight': '16x16/Conv0_up/StyleMod/weight',  # [1024, 512]
    'synthesis.layer4.epilogue.style_mod.dense.wscale.bias': '16x16/Conv0_up/StyleMod/bias',  # [1024]
    'synthesis.layer5.conv.weight': '16x16/Conv1/weight',  # [512, 512, 3, 3]
    'synthesis.layer5.epilogue.apply_noise.weight': '16x16/Conv1/Noise/weight',  # [512]
    'synthesis.layer5.epilogue.bias': '16x16/Conv1/bias',  # [512]
    'synthesis.layer5.epilogue.style_mod.dense.linear.weight': '16x16/Conv1/StyleMod/weight',  # [1024, 512]
    'synthesis.layer5.epilogue.style_mod.dense.wscale.bias': '16x16/Conv1/StyleMod/bias',  # [1024]
    'synthesis.layer6.conv.weight': '32x32/Conv0_up/weight',  # [512, 512, 3, 3]
    'synthesis.layer6.epilogue.apply_noise.weight': '32x32/Conv0_up/Noise/weight',  # [512]
    'synthesis.layer6.epilogue.bias': '32x32/Conv0_up/bias',  # [512]
    'synthesis.layer6.epilogue.style_mod.dense.linear.weight': '32x32/Conv0_up/StyleMod/weight',  # [1024, 512]
    'synthesis.layer6.epilogue.style_mod.dense.wscale.bias': '32x32/Conv0_up/StyleMod/bias',  # [1024]
    'synthesis.layer7.conv.weight': '32x32/Conv1/weight',  # [512, 512, 3, 3]
    'synthesis.layer7.epilogue.apply_noise.weight': '32x32/Conv1/Noise/weight',  # [512]
    'synthesis.layer7.epilogue.bias': '32x32/Conv1/bias',  # [512]
    'synthesis.layer7.epilogue.style_mod.dense.linear.weight': '32x32/Conv1/StyleMod/weight',  # [1024, 512]
    'synthesis.layer7.epilogue.style_mod.dense.wscale.bias': '32x32/Conv1/StyleMod/bias',  # [1024]
    'synthesis.layer8.conv.weight': '64x64/Conv0_up/weight',  # [256, 512, 3, 3]
    'synthesis.layer8.epilogue.apply_noise.weight': '64x64/Conv0_up/Noise/weight',  # [256]
    'synthesis.layer8.epilogue.bias': '64x64/Conv0_up/bias',  # [256]
    'synthesis.layer8.epilogue.style_mod.dense.linear.weight': '64x64/Conv0_up/StyleMod/weight',  # [512, 512]
    'synthesis.layer8.epilogue.style_mod.dense.wscale.bias': '64x64/Conv0_up/StyleMod/bias',  # [512]
    'synthesis.layer9.conv.weight': '64x64/Conv1/weight',  # [256, 256, 3, 3]
    'synthesis.layer9.epilogue.apply_noise.weight': '64x64/Conv1/Noise/weight',  # [256]
    'synthesis.layer9.epilogue.bias': '64x64/Conv1/bias',  # [256]
    'synthesis.layer9.epilogue.style_mod.dense.linear.weight': '64x64/Conv1/StyleMod/weight',  # [512, 512]
    'synthesis.layer9.epilogue.style_mod.dense.wscale.bias': '64x64/Conv1/StyleMod/bias',  # [512]
    'synthesis.layer10.conv.weight': '128x128/Conv0_up/weight',  # [128, 256, 3, 3]
    'synthesis.layer10.epilogue.apply_noise.weight': '128x128/Conv0_up/Noise/weight',  # [128]
    'synthesis.layer10.epilogue.bias': '128x128/Conv0_up/bias',  # [128]
    'synthesis.layer10.epilogue.style_mod.dense.linear.weight': '128x128/Conv0_up/StyleMod/weight',  # [256, 512]
    'synthesis.layer10.epilogue.style_mod.dense.wscale.bias': '128x128/Conv0_up/StyleMod/bias',  # [256]
    'synthesis.layer11.conv.weight': '128x128/Conv1/weight',  # [128, 128, 3, 3]
    'synthesis.layer11.epilogue.apply_noise.weight': '128x128/Conv1/Noise/weight',  # [128]
    'synthesis.layer11.epilogue.bias': '128x128/Conv1/bias',  # [128]
    'synthesis.layer11.epilogue.style_mod.dense.linear.weight': '128x128/Conv1/StyleMod/weight',  # [256, 512]
    'synthesis.layer11.epilogue.style_mod.dense.wscale.bias': '128x128/Conv1/StyleMod/bias',  # [256]
    'synthesis.layer12.conv.weight': '256x256/Conv0_up/weight',  # [64, 128, 3, 3]
    'synthesis.layer12.epilogue.apply_noise.weight': '256x256/Conv0_up/Noise/weight',  # [64]
    'synthesis.layer12.epilogue.bias': '256x256/Conv0_up/bias',  # [64]
    'synthesis.layer12.epilogue.style_mod.dense.linear.weight': '256x256/Conv0_up/StyleMod/weight',  # [128, 512]
    'synthesis.layer12.epilogue.style_mod.dense.wscale.bias': '256x256/Conv0_up/StyleMod/bias',  # [128]
    'synthesis.layer13.conv.weight': '256x256/Conv1/weight',  # [64, 64, 3, 3]
    'synthesis.layer13.epilogue.apply_noise.weight': '256x256/Conv1/Noise/weight',  # [64]
    'synthesis.layer13.epilogue.bias': '256x256/Conv1/bias',  # [64]
    'synthesis.layer13.epilogue.style_mod.dense.linear.weight': '256x256/Conv1/StyleMod/weight',  # [128, 512]
    'synthesis.layer13.epilogue.style_mod.dense.wscale.bias': '256x256/Conv1/StyleMod/bias',  # [128]
    'synthesis.layer14.conv.weight': '512x512/Conv0_up/weight',  # [32, 64, 3, 3]
    'synthesis.layer14.epilogue.apply_noise.weight': '512x512/Conv0_up/Noise/weight',  # [32]
    'synthesis.layer14.epilogue.bias': '512x512/Conv0_up/bias',  # [32]
    'synthesis.layer14.epilogue.style_mod.dense.linear.weight': '512x512/Conv0_up/StyleMod/weight',  # [64, 512]
    'synthesis.layer14.epilogue.style_mod.dense.wscale.bias': '512x512/Conv0_up/StyleMod/bias',  # [64]
    'synthesis.layer15.conv.weight': '512x512/Conv1/weight',  # [32, 32, 3, 3]
    'synthesis.layer15.epilogue.apply_noise.weight': '512x512/Conv1/Noise/weight',  # [32]
    'synthesis.layer15.epilogue.bias': '512x512/Conv1/bias',  # [32]
    'synthesis.layer15.epilogue.style_mod.dense.linear.weight': '512x512/Conv1/StyleMod/weight',  # [64, 512]
    'synthesis.layer15.epilogue.style_mod.dense.wscale.bias': '512x512/Conv1/StyleMod/bias',  # [64]
    'synthesis.layer16.conv.weight': '1024x1024/Conv0_up/weight',  # [16, 32, 3, 3]
    'synthesis.layer16.epilogue.apply_noise.weight': '1024x1024/Conv0_up/Noise/weight',  # [16]
    'synthesis.layer16.epilogue.bias': '1024x1024/Conv0_up/bias',  # [16]
    'synthesis.layer16.epilogue.style_mod.dense.linear.weight': '1024x1024/Conv0_up/StyleMod/weight',  # [32, 512]
    'synthesis.layer16.epilogue.style_mod.dense.wscale.bias': '1024x1024/Conv0_up/StyleMod/bias',  # [32]
    'synthesis.layer17.conv.weight': '1024x1024/Conv1/weight',  # [16, 16, 3, 3]
    'synthesis.layer17.epilogue.apply_noise.weight': '1024x1024/Conv1/Noise/weight',  # [16]
    'synthesis.layer17.epilogue.bias': '1024x1024/Conv1/bias',  # [16]
    'synthesis.layer17.epilogue.style_mod.dense.linear.weight': '1024x1024/Conv1/StyleMod/weight',  # [32, 512]
    'synthesis.layer17.epilogue.style_mod.dense.wscale.bias': '1024x1024/Conv1/StyleMod/bias',  # [32]
    'synthesis.output0.conv.weight': 'ToRGB_lod8/weight',  # [3, 512, 1, 1]
    'synthesis.output0.bias': 'ToRGB_lod8/bias',  # [3]
    'synthesis.output1.conv.weight': 'ToRGB_lod7/weight',  # [3, 512, 1, 1]
    'synthesis.output1.bias': 'ToRGB_lod7/bias',  # [3]
    'synthesis.output2.conv.weight': 'ToRGB_lod6/weight',  # [3, 512, 1, 1]
    'synthesis.output2.bias': 'ToRGB_lod6/bias',  # [3]
    'synthesis.output3.conv.weight': 'ToRGB_lod5/weight',  # [3, 512, 1, 1]
    'synthesis.output3.bias': 'ToRGB_lod5/bias',  # [3]
    'synthesis.output4.conv.weight': 'ToRGB_lod4/weight',  # [3, 256, 1, 1]
    'synthesis.output4.bias': 'ToRGB_lod4/bias',  # [3]
    'synthesis.output5.conv.weight': 'ToRGB_lod3/weight',  # [3, 128, 1, 1]
    'synthesis.output5.bias': 'ToRGB_lod3/bias',  # [3]
    'synthesis.output6.conv.weight': 'ToRGB_lod2/weight',  # [3, 64, 1, 1]
    'synthesis.output6.bias': 'ToRGB_lod2/bias',  # [3]
    'synthesis.output7.conv.weight': 'ToRGB_lod1/weight',  # [3, 32, 1, 1]
    'synthesis.output7.bias': 'ToRGB_lod1/bias',  # [3]
    'synthesis.output8.conv.weight': 'ToRGB_lod0/weight',  # [3, 16, 1, 1]
    'synthesis.output8.bias': 'ToRGB_lod0/bias',  # [3]
}
# pylint: enable=line-too-long

# Minimal resolution for `auto` fused-scale strategy.
_AUTO_FUSED_SCALE_MIN_RES = 128


class StyleGANGeneratorModel(nn.Module):
  """Defines the generator module in StyleGAN.

  Note that the generated images are with RGB color channels.
  """

  def __init__(self,
               resolution=1024,
               w_space_dim=512,
               fused_scale='auto',
               output_channels=3,
               truncation_psi=0.7,
               truncation_layers=8,
               randomize_noise=False):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the final output image. (default: 1024)
      w_space_dim: The dimension of the disentangled latent vectors, w.
        (default: 512)
      fused_scale: If set as `True`, `conv2d_transpose` is used for upscaling.
        If set as `False`, `upsample + conv2d` is used for upscaling. If set as
        `auto`, `upsample + conv2d` is used for bottom layers until resolution
        reaches 128. (default: `auto`)
      output_channels: Number of channels of output image. (default: 3)
      truncation_psi: Style strength multiplier for the truncation trick.
        `None` or `1.0` indicates no truncation. (default: 0.7)
      truncation_layers: Number of layers for which to apply the truncation
        trick. `None` indicates no truncation. (default: 8)
      randomize_noise: Whether to add random noise for each convolutional layer.
        (default: False)

    Raises:
      ValueError: If the input `resolution` is not supported.
    """
    super().__init__()
    self.resolution = resolution
    self.w_space_dim = w_space_dim
    self.fused_scale = fused_scale
    self.output_channels = output_channels
    self.truncation_psi = truncation_psi
    self.truncation_layers = truncation_layers
    self.randomize_noise = randomize_noise

    self.mapping = MappingModule(final_space_dim=self.w_space_dim)
    self.truncation = TruncationModule(resolution=self.resolution,
                                       w_space_dim=self.w_space_dim,
                                       truncation_psi=self.truncation_psi,
                                       truncation_layers=self.truncation_layers)
    self.synthesis = SynthesisModule(resolution=self.resolution,
                                     fused_scale=self.fused_scale,
                                     output_channels=self.output_channels,
                                     randomize_noise=self.randomize_noise)

    self.pth_to_tf_var_mapping = {}
    for pth_var_name, tf_var_name in _STYLEGAN_PTH_VARS_TO_TF_VARS.items():
      if 'Conv0_up' in tf_var_name:
        res = int(tf_var_name.split('x')[0])
        if ((self.fused_scale is True) or
            (self.fused_scale == 'auto' and res >= _AUTO_FUSED_SCALE_MIN_RES)):
          pth_var_name = pth_var_name.replace('conv.weight', 'weight')
      self.pth_to_tf_var_mapping[pth_var_name] = tf_var_name

  def forward(self, z):
    w = self.mapping(z)
    w = self.truncation(w)
    x = self.synthesis(w)
    return x


class MappingModule(nn.Sequential):
  """Implements the latent space mapping module used in StyleGAN.

  Basically, this module executes several dense layers in sequence.
  """

  def __init__(self,
               normalize_input=True,
               input_space_dim=512,
               hidden_space_dim=512,
               final_space_dim=512,
               num_layers=8):
    sequence = OrderedDict()

    def _add_layer(layer, name=None):
      name = name or f'dense{len(sequence) + (not normalize_input) - 1}'
      sequence[name] = layer

    if normalize_input:
      _add_layer(PixelNormLayer(), name='normalize')
    for i in range(num_layers):
      in_dim = input_space_dim if i == 0 else hidden_space_dim
      out_dim = final_space_dim if i == (num_layers - 1) else hidden_space_dim
      _add_layer(DenseBlock(in_dim, out_dim))
    super().__init__(sequence)

  def forward(self, x):
    if len(x.shape) != 2:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'noise_dim], but {x.shape} received!')
    return super().forward(x)


class TruncationModule(nn.Module):
  """Implements the truncation module used in StyleGAN."""

  def __init__(self,
               resolution=1024,
               w_space_dim=512,
               truncation_psi=0.7,
               truncation_layers=8):
    super().__init__()

    self.num_layers = int(np.log2(resolution)) * 2 - 2
    self.w_space_dim = w_space_dim
    if truncation_psi is not None and truncation_layers is not None:
      self.use_truncation = True
    else:
      self.use_truncation = False
      truncation_psi = 1.0
      truncation_layers = 0
    self.register_buffer('w_avg', torch.zeros(w_space_dim))
    layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
    coefs = np.ones_like(layer_idx, dtype=np.float32)
    coefs[layer_idx < truncation_layers] *= truncation_psi
    self.register_buffer('truncation', torch.from_numpy(coefs))

  def forward(self, w):
    if len(w.shape) == 2:
      w = w.view(-1, 1, self.w_space_dim).repeat(1, self.num_layers, 1)
    if self.use_truncation:
      w_avg = self.w_avg.view(1, 1, self.w_space_dim)
      w = w_avg + (w - w_avg) * self.truncation
    return w


class SynthesisModule(nn.Module):
  """Implements the image synthesis module used in StyleGAN.

  Basically, this module executes several convolutional layers in sequence.
  """

  def __init__(self,
               resolution=1024,
               fused_scale='auto',
               output_channels=3,
               randomize_noise=False):
    super().__init__()

    try:
      self.channels = _RESOLUTIONS_TO_CHANNELS[resolution]
    except KeyError:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: '
                       f'{list(_RESOLUTIONS_TO_CHANNELS)}.')
    assert len(self.channels) == int(np.log2(resolution))

    for block_idx in range(1, len(self.channels)):
      if block_idx == 1:
        self.add_module(
            f'layer{2 * block_idx - 2}',
            FirstConvBlock(in_channels=self.channels[block_idx - 1],
                           randomize_noise=randomize_noise))
      else:
        self.add_module(
            f'layer{2 * block_idx - 2}',
            UpConvBlock(layer_idx=2 * block_idx - 2,
                        in_channels=self.channels[block_idx - 1],
                        out_channels=self.channels[block_idx],
                        randomize_noise=randomize_noise,
                        fused_scale=fused_scale))
      self.add_module(
          f'layer{2 * block_idx - 1}',
          ConvBlock(layer_idx=2 * block_idx - 1,
                    in_channels=self.channels[block_idx],
                    out_channels=self.channels[block_idx],
                    randomize_noise=randomize_noise))
      self.add_module(
          f'output{block_idx - 1}',
          LastConvBlock(in_channels=self.channels[block_idx],
                        out_channels=output_channels))

    self.upsample = ResolutionScalingLayer()
    self.lod = nn.Parameter(torch.zeros(()))

  def forward(self, w):
    lod = self.lod.cpu().tolist()
    x = self.layer0(w[:, 0])
    for block_idx in range(1, len(self.channels)):
      if block_idx + lod < len(self.channels):
        layer_idx = 2 * block_idx - 2
        if layer_idx == 0:
          x = self.__getattr__(f'layer{layer_idx}')(w[:, layer_idx])
        else:
          x = self.__getattr__(f'layer{layer_idx}')(x, w[:, layer_idx])
        layer_idx = 2 * block_idx - 1
        x = self.__getattr__(f'layer{layer_idx}')(x, w[:, layer_idx])
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


class InstanceNormLayer(nn.Module):
  """Implements instance normalization layer."""

  def __init__(self, epsilon=1e-8):
    super().__init__()
    self.epsilon = epsilon

  def forward(self, x):
    if len(x.shape) != 4:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'num_channels, height, width], but {x.shape} received!')
    x = x - torch.mean(x, dim=[2, 3], keepdim=True)
    x = x / torch.sqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) +
                       self.epsilon)
    return x


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


class BlurLayer(nn.Module):
  """Implements the blur layer used in StyleGAN."""

  def __init__(self,
               channels,
               kernel=(1, 2, 1),
               normalize=True,
               flip=False):
    super().__init__()
    kernel = np.array(kernel, dtype=np.float32).reshape(1, 3)
    kernel = kernel.T.dot(kernel)
    if normalize:
      kernel /= np.sum(kernel)
    if flip:
      kernel = kernel[::-1, ::-1]
    kernel = kernel.reshape(3, 3, 1, 1)
    kernel = np.tile(kernel, [1, 1, channels, 1])
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    self.register_buffer('kernel', torch.from_numpy(kernel))
    self.channels = channels

  def forward(self, x):
    return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class NoiseApplyingLayer(nn.Module):
  """Implements the noise applying layer used in StyleGAN."""

  def __init__(self, layer_idx, channels, randomize_noise=False):
    super().__init__()
    self.randomize_noise = randomize_noise
    self.res = 2**(layer_idx // 2 + 2)
    self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
    self.weight = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    if len(x.shape) != 4:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'num_channels, height, width], but {x.shape} received!')
    if self.randomize_noise:
      noise = torch.randn(x.shape[0], 1, self.res, self.res).to(x)
    else:
      noise = self.noise
    return x + noise * self.weight.view(1, -1, 1, 1)


class StyleModulationLayer(nn.Module):
  """Implements the style modulation layer used in StyleGAN."""

  def __init__(self, channels, w_space_dim=512):
    super().__init__()
    self.channels = channels
    self.dense = DenseBlock(in_features=w_space_dim,
                            out_features=channels*2,
                            wscale_gain=1.0,
                            wscale_lr_multiplier=1.0,
                            activation_type='linear')

  def forward(self, x, w):
    if len(w.shape) != 2:
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'num_channels], but {x.shape} received!')
    style = self.dense(w)
    style = style.view(-1, 2, self.channels, 1, 1)
    return x * (style[:, 0] + 1) + style[:, 1]


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  Note that, the weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
  layer), and only scaled with a constant number , which is not trainable, in
  this layer. However, the bias variable is trainable in this layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0),
               lr_multiplier=1.0):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in) * lr_multiplier
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.lr_multiplier = lr_multiplier

  def forward(self, x):
    if len(x.shape) == 4:
      return x * self.scale + self.bias.view(1, -1, 1, 1) * self.lr_multiplier
    if len(x.shape) == 2:
      return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier
    raise ValueError(f'The input tensor should be with shape [batch_size, '
                     f'num_channels, height, width], or [batch_size, '
                     f'num_channels], but {x.shape} received!')


class EpilogueBlock(nn.Module):
  """Implements the epilogue block of each conv block."""

  def __init__(self,
               layer_idx,
               channels,
               randomize_noise=False,
               normalization_fn='instance'):
    super().__init__()
    self.apply_noise = NoiseApplyingLayer(layer_idx, channels, randomize_noise)
    self.bias = nn.Parameter(torch.zeros(channels))
    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if normalization_fn == 'pixel':
      self.norm = PixelNormLayer()
    elif normalization_fn == 'instance':
      self.norm = InstanceNormLayer()
    else:
      raise NotImplementedError(f'Not implemented normalization function: '
                                f'{normalization_fn}!')
    self.style_mod = StyleModulationLayer(channels)

  def forward(self, x, w):
    x = self.apply_noise(x)
    x = x + self.bias.view(1, -1, 1, 1)
    x = self.activate(x)
    x = self.norm(x)
    x = self.style_mod(x, w)
    return x


class FirstConvBlock(nn.Module):
  """Implements the first convolutional block used in StyleGAN.

  Basically, this block starts from a const input, which is `ones(512, 4, 4)`.
  """

  def __init__(self, in_channels, randomize_noise=False):
    super().__init__()
    self.first_layer = nn.Parameter(torch.ones(1, in_channels, 4, 4))
    self.epilogue = EpilogueBlock(layer_idx=0,
                                  channels=in_channels,
                                  randomize_noise=randomize_noise)

  def forward(self, w):
    x = self.first_layer.repeat(w.shape[0], 1, 1, 1)
    x = self.epilogue(x, w)
    return x


class UpConvBlock(nn.Module):
  """Implements the convolutional block used in StyleGAN.

  Basically, this block is used as the first convolutional block for each
  resolution, which will execute upsampling.
  """

  def __init__(self,
               layer_idx,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               fused_scale='auto',
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               randomize_noise=False):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      fused_scale: Whether to fuse `upsample` and `conv2d` together, resulting
        in `conv2d_transpose`.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      randomize_noise: Whether to add random noise.

    Raises:
      ValueError: If the block is not applied to the first block for a
        particular resolution. Or `fused_scale` does not belong to [True, False,
        `auto`].
    """
    super().__init__()
    if layer_idx % 2 == 1:
      raise ValueError(f'This block is implemented as the first block of each '
                       f'resolution, but is applied to layer {layer_idx}!')
    if fused_scale not in [True, False, 'auto']:
      raise ValueError(f'`fused_scale` can only be [True, False, `auto`], '
                       f'but {fused_scale} received!')

    cur_res = 2 ** (layer_idx // 2 + 2)
    if fused_scale == 'auto':
      self.fused_scale = (cur_res >= _AUTO_FUSED_SCALE_MIN_RES)
    else:
      self.fused_scale = fused_scale

    if self.fused_scale:
      self.weight = nn.Parameter(
          torch.randn(kernel_size, kernel_size, in_channels, out_channels))

    else:
      self.upsample = ResolutionScalingLayer()
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            bias=add_bias)

    fan_in = in_channels * kernel_size * kernel_size
    self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
    self.blur = BlurLayer(channels=out_channels)
    self.epilogue = EpilogueBlock(layer_idx=layer_idx,
                                  channels=out_channels,
                                  randomize_noise=randomize_noise)

  def forward(self, x, w):
    if self.fused_scale:
      kernel = self.weight * self.scale
      kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
      kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                kernel[1:, :-1] + kernel[:-1, :-1])
      kernel = kernel.permute(2, 3, 0, 1)
      x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
    else:
      x = self.upsample(x)
      x = self.conv(x) * self.scale
    x = self.blur(x)
    x = self.epilogue(x, w)
    return x


class ConvBlock(nn.Module):
  """Implements the convolutional block used in StyleGAN.

  Basically, this block is used as the second convolutional block for each
  resolution.
  """

  def __init__(self,
               layer_idx,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               randomize_noise=False):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels (kernels) of the output tensor.
      kernel_size: Size of the convolutional kernel.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      randomize_noise: Whether to add random noise.

    Raises:
      ValueError: If the block is not applied to the second block for a
        particular resolution.
    """
    super().__init__()
    if layer_idx % 2 == 0:
      raise ValueError(f'This block is implemented as the second block of each '
                       f'resolution, but is applied to layer {layer_idx}!')

    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation,
                          groups=1,
                          bias=add_bias)
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
    self.epilogue = EpilogueBlock(layer_idx=layer_idx,
                                  channels=out_channels,
                                  randomize_noise=randomize_noise)

  def forward(self, x, w):
    x = self.conv(x) * self.scale
    x = self.epilogue(x, w)
    return x


class LastConvBlock(nn.Module):
  """Implements the last convolutional block used in StyleGAN.

  Basically, this block converts the final feature map to RGB image.
  """

  def __init__(self, in_channels, out_channels=3):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          bias=False)
    self.scale = 1 / np.sqrt(in_channels)
    self.bias = nn.Parameter(torch.zeros(3))

  def forward(self, x):
    x = self.conv(x) * self.scale
    x = x + self.bias.view(1, -1, 1, 1)
    return x


class DenseBlock(nn.Module):
  """Implements the dense block used in StyleGAN.

  Basically, this block executes fully-connected layer, weight-scale layer,
  and activation layer in sequence.
  """

  def __init__(self,
               in_features,
               out_features,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=0.01,
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_features: Number of channels of the input tensor fed into this block.
      out_features: Number of channels of the output tensor.
      add_bias: Whether to add bias onto the fully-connected result.
      wscale_gain: The gain factor for `wscale` layer.
      wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
        layer.
      activation_type: Type of activation function. Support `linear` and
        `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()
    self.linear = nn.Linear(in_features=in_features,
                            out_features=out_features,
                            bias=add_bias)
    self.wscale = WScaleLayer(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=1,
                              gain=wscale_gain,
                              lr_multiplier=wscale_lr_multiplier)
    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    x = self.linear(x)
    x = self.wscale(x)
    x = self.activate(x)
    return x
