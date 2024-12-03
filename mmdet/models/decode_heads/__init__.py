from .daformer_head import DAFormerHead
from .daformer_head_devil import DAFormerHeadDevil
from .daformer_head_devil2 import DAFormerHeadDevil2
from .daformer_head_devil3 import DAFormerHeadDevil3
from .daformer_head_devil4 import DAFormerHeadDevil4
from .daformer_head_devil6 import DAFormerHeadDevil6
from .daformer_head_devil_dacTextEmbed import DAFormerHeadDevilDacsJustThings
from .daformer_head_devil4_copy import DAFormerHeadDevil4Copy
from .aspp_head import ASPPHead
from .isa_head import ISAHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .dlv2_head import DLV2Head
from .fpn_head import FPNHead
__all__ =   [
                'FPNHead',
                'DAFormerHead',
                'DAFormerHeadDevil',
                'DAFormerHeadDevil2',
                'ASPPHead',
                'ISAHead',
                'SegFormerHead',
                'DepthwiseSeparableASPPHead',
                'DLV2Head',
            ]