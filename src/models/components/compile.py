"""torch.compile helpers for SSFM adapter backbones.

The XL/2 backbone is a stack of identical :class:`DiTBlock` modules, so the
useful compilation unit is *one block*, not the whole two-time field. Compiling
the field as a single graph under ``mode='reduce-overhead'`` captures all 28
blocks into one CUDA graph, which is re-captured whenever the batch size, the
grad mode, or the tensor addresses change. The discrete ESD/LSD objectives hit
the same field with three different configurations per step (diagonal no-grad,
off-interval no-grad, off-interval with grad), so that graph thrashes.

Compiling each block instead gives inductor one small, highly repeated region
to fuse and lets every caller -- student, frozen teacher, and EMA -- share the
same compiled artifact.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

COMPILE_MODE = 'default'


def compile_blocks(module: nn.Module) -> None:
    """Compile each transformer block of ``module`` in place.

    ``nn.Module.compile`` wraps the module's ``forward`` without re-parenting it,
    so parameter and buffer names keep their original spelling and the EMA
    shadow copy still matches by name.

    :param module: A backbone exposing a ``blocks`` sequence.
    """
    blocks = getattr(module, 'blocks', None)
    if blocks is None:
        raise TypeError('Block-level compilation requires a backbone with .blocks.')
    for block in blocks:
        block.compile(mode=COMPILE_MODE, dynamic=False)


def compile_adapter_backbone(adapter: Any, *, pretrained: bool, field: bool) -> None:
    """Compile the shared block stack backing every adapter prediction path.

    Both flags are accepted for call-site compatibility. They no longer select
    distinct graphs: the pretrained, diagonal, and field paths all run the same
    blocks, so compiling the stack once serves all of them.

    :param adapter: The SSFM adapter owning a ``base`` backbone.
    :param pretrained: Whether the frozen teacher path is used.
    :param field: Whether the two-time field path is used.
    """
    if not (pretrained or field):
        return
    compile_blocks(adapter.base)
