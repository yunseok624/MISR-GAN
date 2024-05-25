import torch

def has_black_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    black_pixels = (channel_sum.view(-1) == 0).any()

    return black_pixels

def has_white_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    white_pixels = (channel_sum.view(-1) == 1).any()

    return white_pixels

def has_cloud_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    cloud_pixels = (channel_sum.view(-1) > 0.5).any()

    return cloud_pixels