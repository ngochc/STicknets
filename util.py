import torch


def get_device(gpu_id=0):
  """
  Determine the device to use for PyTorch operations.

  Args:
    gpu_id (int): GPU ID to use. Set to -1 to use CPU.

  Returns:
    torch.device: The device to use (cuda, mps, or cpu)
  """
  if gpu_id >= 0 and torch.cuda.is_available():
    return torch.device('cuda:{}'.format(gpu_id))
  elif torch.backends.mps.is_available():
    return torch.device('mps')
  else:
    return torch.device('cpu')
