import torch
import torch.nn as nn
from .utils import manual_cast


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        target_dtype = self.weight.dtype if self.weight is not None else torch.float32
        x = manual_cast(x, target_dtype)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        target_dtype = self.weight.dtype if self.weight is not None else torch.float32
        x = manual_cast(x, target_dtype)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    