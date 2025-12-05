"""
Compatibility wrapper for VectorQuantizer to handle both naming conventions.
This allows the code to use n_embed/embed_dim while VectorQuantizer2 uses n_e/e_dim.
"""

from .quantize import VectorQuantizer2 as _VectorQuantizer2


class VectorQuantizer(_VectorQuantizer2):
    """Wrapper that accepts both n_embed/embed_dim and n_e/e_dim parameter names."""
    
    def __init__(self, n_embed=None, embed_dim=None, n_e=None, e_dim=None, beta=0.25, **kwargs):
        # Support both naming conventions
        if n_embed is not None and n_e is None:
            n_e = n_embed
        if embed_dim is not None and e_dim is None:
            e_dim = embed_dim
        
        # Call parent with correct parameter names
        super().__init__(n_e=n_e, e_dim=e_dim, beta=beta, **kwargs)


# Also export VectorQuantizer2 for compatibility
VectorQuantizer2 = VectorQuantizer
