
import torch
import torch.nn as nn

from .se_attention import SE
from .common import Classifier, conv1x1_block, conv3x3_block, conv3x3_dw_block_all


class LWO(torch.nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.dwLWO = conv3x3_dw_block_all(channels=in_channels, stride=stride)
    self.PwOut = conv1x1_block(
      in_channels=in_channels, out_channels=out_channels)

  def forward(self, x):
    x = self.dwLWO(x)
    x = self.PwOut(x)
    return x


class FR_PDP_block(torch.nn.Module):
  """
  FR_PDP_block for TickNet.
  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    stride (int): Stride for depthwise. convolution.
  """

  def __init__(self,
         in_channels,
         out_channels,
         stride):

    super().__init__()
    self.Pw1 = conv1x1_block(
      in_channels=in_channels, out_channels=in_channels, use_bn=False, activation=None)
    self.Dw = conv3x3_dw_block_all(channels=in_channels, stride=stride)
    self.Pw2 = conv1x1_block(
      in_channels=in_channels, out_channels=out_channels, groups=1)
    self.PwR = conv1x1_block(
      in_channels=in_channels, out_channels=out_channels, stride=stride)

    self.stride = stride
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.SE = SE(out_channels, 16)

  def forward(self, x):
    residual = x
    x = self.Pw1(x)
    x = self.Dw(x)
    x = self.Pw2(x)
    x = self.SE(x)
    if self.stride == 1 and self.in_channels == self.out_channels:
      x = x + residual
    else:
      residual = self.PwR(residual)
      x = x + residual
    return x


class SpatialTickNet(nn.Module):
  """
  Class for constructing SpatialTickNet, enhancing TickNet for spatial feature learning.
  Args:
    num_classes (int): Number of output classes.
    init_conv_channels (int): Initial convolution channels.
    init_conv_stride (int): Initial convolution stride.
    channels (list): List of channel configurations for each stage.
    strides (list): List of stride values for each stage.
    in_channels (int): Number of input channels.
    in_size (tuple): Input image size.
    use_data_batchnorm (bool): Whether to use batch normalization on input data.
  """

  def __init__(self,
         num_classes: int,
         init_conv_channels: int,
         init_conv_stride: int,
         channels: list,
         strides: list,
         in_channels: int = 3,
         in_size: tuple = (224, 224),
         use_data_batchnorm: bool = True,
         use_lightweight_optimization: bool = False
         ):

    super().__init__()
    self.use_data_batchnorm = use_data_batchnorm
    self.in_size = in_size
    self.use_lightweight_optimization = use_lightweight_optimization

    self.backbone = torch.nn.Sequential()

    # data batchnorm
    if self.use_data_batchnorm:
      self.backbone.add_module(
        "data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

    # init conv
    self.backbone.add_module("init_conv", conv3x3_block(
      in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

    # Validate input
    if len(channels) != len(strides):
      raise ValueError("channels and strides must have the same length")
    # stages
    in_channels = init_conv_channels
    in_channels = self.build_stages(
      in_channels, channels, strides, self.use_lightweight_optimization)

    self.final_conv_channels = 1024
    self.backbone.add_module("final_conv", conv1x1_block(
      in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
    self.backbone.add_module(
      "dropout1", torch.nn.Dropout2d(0.2))  # with dropout
    self.backbone.add_module(
      "global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
    self.backbone.add_module(
      "dropout2", torch.nn.Dropout2d(0.2))  # with dropout

    in_channels = self.final_conv_channels
    # classifier
    self.classifier = Classifier(
      in_channels=in_channels, num_classes=num_classes)

    self.init_params()

  def init_params(self):
    # backbone
    for _, module in self.backbone.named_modules():
      if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
          torch.nn.init.constant_(module.bias, 0)
    # classifier
    self.classifier.init_params()

  def forward(self, x):
    x = self.backbone(x)
    x = self.classifier(x)
    return x

  def build_stages(self, in_channels: int, channels: list, strides: list, use_lightweight_optimization: bool = False) -> int:
    """
    Build the stages of the backbone network.
    Args:
      in_channels (int): Initial input channels.
      channels (list): List of channel configurations for each stage.
      strides (list): List of stride values for each stage.
      use_lightweight_optimization (bool): Whether to use LWO blocks when conditions are met.
    Returns:
      int: Number of output channels after all stages.
    """
    for stage_id, stage_channels in enumerate(channels):
      stage = torch.nn.Sequential()
      for unit_id, unit_channels in enumerate(stage_channels):
        stride = strides[stage_id] if unit_id == 0 else 1
        if use_lightweight_optimization and in_channels == 256 and unit_channels == 128:
          unit = LWO(in_channels=in_channels, out_channels=unit_channels, stride=stride)
          stage.add_module("LWO{}".format(unit_id + 1), unit)
        else:
          unit = FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride)
          stage.add_module("unit{}".format(unit_id + 1), unit)
        in_channels = unit_channels
      self.backbone.add_module("stage{}".format(stage_id + 1), stage)
    return in_channels

###
# %% model definitions
###


def build_STickNet(num_classes: int, typesize: str = 'small', cifar: bool = False, use_lightweight_optimization: bool = False) -> SpatialTickNet:
  """
  Build a SpatialTickNet model.
  Args:
    num_classes (int): Number of output classes.
    typesize (str): Model size type ('basic', 'small', 'large').
    cifar (bool): Whether to use CIFAR input size.
    use_lightweight_optimization (bool): Whether to enable LWO blocks for 256->128 channel transitions.
  Returns:
    SpatialTickNet: The constructed model.
  """
  init_conv_channels = 32
  if typesize == 'basic':
    channels = [[256], [128, 64], [128], [256], [512]]
  elif typesize == 'small':
    channels = [[256], [128, 64, 128], [
      256, 512, 256, 128], [64, 128, 256], [512]]
  elif typesize == 'large':
    channels = [[256, 128], [64, 128, 256], [512, 256, 128, 64, 128, 256, 512], [256, 128, 64, 128, 256], [512]]
  else:
    raise ValueError("typesize must be 'basic', 'small', or 'large'")

  if cifar:
    in_size = (32, 32)
    init_conv_stride = 1
    strides = [1, 1, 2, 2, 2]
  else:
    in_size = (224, 224)
    init_conv_stride = 2
    strides = [2, 1, 2, 2, 2]  # for all

  return SpatialTickNet(
    num_classes=num_classes,
    init_conv_channels=init_conv_channels,
    init_conv_stride=init_conv_stride,
    channels=channels,
    strides=strides,
    in_size=in_size,
    use_lightweight_optimization=use_lightweight_optimization)
