"""DMF anchor losses for the released experiments (ImageNet-256 fine-tuning).

Each loss pairs a diagonal anchor (a stopped teacher field on the EI skeleton)
with an off-diagonal objective: the semigroup PSD anchor and the discrete-AYF
stopped-substep ESD/LSD variants (:mod:`.dmf_ei_euler`), and the WZ-iMF
interval-average objective with its AYF-LMD Lagrangian counterpart
(:mod:`.dmf_wz_imf`). :class:`DMFAblationPSDLoss` (:mod:`.dmf_ablation`) is the
semigroup loss for the SSFM -> iSSFM ablation ladder (learned noise / no lift).
"""

from .dmf_ablation import DMFAblationPSDLoss
from .dmf_ei_euler import (
    DMFEIEulerAnchorDiscreteESDLoss,
    DMFEIEulerAnchorDiscreteLSDLoss,
    DMFEIEulerAnchorPSDLoss,
)
from .dmf_wz_imf import DMFWZEIIMFAnchorLoss, DMFWZEILMDAnchorLoss

__all__ = [
    'DMFEIEulerAnchorPSDLoss',
    'DMFAblationPSDLoss',
    'DMFEIEulerAnchorDiscreteESDLoss',
    'DMFEIEulerAnchorDiscreteLSDLoss',
    'DMFWZEIIMFAnchorLoss',
    'DMFWZEILMDAnchorLoss',
]
