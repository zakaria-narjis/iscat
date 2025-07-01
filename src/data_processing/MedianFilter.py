import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from scipy.ndimage import median_filter
import numpy as np

class MemoryEfficientMedianPool2d(nn.Module):
    """Memory-efficient median pooling that processes in chunks."""
    
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, chunk_size=32):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
        self.chunk_size = chunk_size
    
    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_padded = F.pad(x, self._padding(x), mode='reflect')
        
        # Process in chunks to reduce memory usage
        results = []
        for i in range(0, B, self.chunk_size):
            chunk = x_padded[i:i+self.chunk_size]
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    def _process_chunk(self, x):
        # Use unfold on smaller chunks
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class SeparableMedianFilter(nn.Module):
    """Approximates 2D median with separable 1D medians (much faster, less memory)."""
    
    def __init__(self, kernel_size=31):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def forward(self, x):
        # Apply 1D median filter horizontally then vertically
        B, C, H, W = x.shape
        
        # Horizontal median
        x_padded = F.pad(x, (self.padding, self.padding, 0, 0), mode='reflect')
        x_unfolded = x_padded.unfold(3, self.kernel_size, 1)
        x_h_median = x_unfolded.median(dim=-1)[0]
        
        # Vertical median
        x_h_padded = F.pad(x_h_median, (0, 0, self.padding, self.padding), mode='reflect')
        x_v_unfolded = x_h_padded.unfold(2, self.kernel_size, 1)
        x_median = x_v_unfolded.median(dim=-1)[0]
        
        return x_median

def cpu_median_filter(image_stack, kernel_size):
    """CPU fallback using scipy for very large images."""      
    result = np.zeros_like(image_stack)
    for i in range(image_stack.shape[0]):
        for c in range(image_stack.shape[1]):
            result[i, c] = median_filter(image_stack[i, c], size=kernel_size, mode='reflect')
    return result