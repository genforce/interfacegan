# python3.7
"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np
import pickle
from PIL import Image

from typing import List, Optional, Tuple, Union

import torch

from . import model_settings
from .stylegan3_official_network import StyleGAN3GeneratorModel
from .base_generator import BaseGenerator

__all__ = ['StyleGANGenerator']

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

class StyleGAN3Generator(BaseGenerator):
  """Defines the generator class of StyleGAN.

  Different from conventional GAN, StyleGAN introduces a disentangled latent
  space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
  the disentangled latent code, w, is fed into each convolutional layer to
  modulate the `style` of the synthesis through AdaIN (Adaptive Instance
  Normalization) layer. Normally, the w's fed into all layers are the same. But,
  they can actually be different to make different layers get different styles.
  Accordingly, an extended space (i.e. W+ space) is used to gather all w's
  together. Taking the official StyleGAN model trained on FF-HQ dataset as an
  instance, there are
  (1) Z space, with dimension (512,)
  (2) W space, with dimension (512,)
  (3) W+ space, with dimension (18, 512)
  """

  def __init__(self, model_name, logger=None):
    self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
    self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
    self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
    self.model_specific_vars = ['truncation.truncation']
    super().__init__(model_name, logger)
    self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
    assert self.gan_type == 'stylegan3'

  def build(self):
    self.check_attr('w_space_dim')
    self.check_attr('fused_scale')
    self.model = StyleGAN3GeneratorModel(
        img_resolution=self.resolution,
        w_dim=self.w_space_dim,
        z_dim=self.latent_space_dim,
        c_dim=self.c_space_dim,
        img_channels=3
        )


  def load(self):
    self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
    with open(self.model_path, 'rb') as f:
      self.model = pickle.load(f)['G_ema']
    self.logger.info(f'Successfully loaded!')
    # self.lod = self.model.synthesis.lod.to(self.cpu_device).tolist()
    # self.logger.info(f'  `lod` of the loaded model is {self.lod}.')


  def sample(self, num, latent_space_type='Z'):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.
      latent_space_type: Type of latent space from which to sample latent code.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

    Returns:
      A `numpy.ndarray` as sampled latend codes.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = np.random.randn(num, self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = np.random.randn(num, self.w_space_dim)
    elif latent_space_type == 'WP':
      latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def preprocess(self, latent_codes, latent_space_type='Z'):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
      latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
    elif latent_space_type == 'WP':
      latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def easy_sample(self, num, latent_space_type='Z'):
    return self.sample(num, latent_space_type)

  def synthesize(self,
                 latent_codes,
                 latent_space_type='Z',
                 generate_style=False,
                 generate_image=True):
    """Synthesizes images with given latent codes.

    One can choose whether to generate the layer-wise style codes.

    Args:
      latent_codes: Input latent codes for image synthesis.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
      generate_style: Whether to generate the layer-wise style codes. (default:
        False)
      generate_image: Whether to generate the final image synthesis. (default:
        True)

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}
    translate = (0,0)
    rotate=0.5
    latent_space_type = latent_space_type.upper()
    z = torch.from_numpy(latent_codes).to(self.run_device)
    label = torch.ones([1, self.c_space_dim]).to(self.run_device)

    if hasattr(self.model.synthesis, 'input'):
      m = make_transform(translate, rotate)
      m = np.linalg.inv(m)
      self.model.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = self.model(z, label)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.cpu().numpy()

    results['image'] = img
    results['z'] = latent_codes
    return results
