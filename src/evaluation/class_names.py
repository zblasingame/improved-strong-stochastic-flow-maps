"""ImageNet class names and caption prompts for text-conditioned reward models.

The names are OpenAI CLIP's cleaned ImageNet-1k list (``notebooks/
Prompt_Engineering_for_ImageNet.ipynb``), vendored verbatim, index-aligned with
the standard ILSVRC-2012 label ids. The list is kept verbatim for comparability,
so two pairs of classes share a caption: 657/744 (``missile``) and 836/837
(``sunglasses``).

Stdlib-only: imported by the isolated preference-scoring environment.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

CLASS_NAMES_PATH = Path(__file__).with_name('imagenet_class_names.json')
NUM_CLASSES = 1000
PROMPT_TEMPLATE = 'A photo of {article} {name}'

# Vowel-initial words pronounced with a consonant sound, and vice versa.
_CONSONANT_SOUND_PREFIXES = ('one', 'eu', 'uni', 'use')
_VOWEL_SOUND_PREFIXES = ('hour', 'honest', 'heir')


@cache
def load_class_names() -> tuple[str, ...]:
    """Return the 1000 ImageNet class names indexed by label id."""
    names = tuple(json.loads(CLASS_NAMES_PATH.read_text()))
    if len(names) != NUM_CLASSES:
        raise ValueError(f'expected {NUM_CLASSES} class names, found {len(names)}')
    return names


def indefinite_article(name: str) -> str:
    """Return ``'a'`` or ``'an'`` for ``name`` using a first-sound heuristic."""
    lowered = name.lower()
    if lowered.startswith(_VOWEL_SOUND_PREFIXES):
        return 'an'
    if lowered.startswith(_CONSONANT_SOUND_PREFIXES):
        return 'a'
    return 'an' if lowered[:1] in 'aeiou' else 'a'


def make_prompt(name: str) -> str:
    """Build the caption ``'A photo of a/an {name}'`` for one class name."""
    return PROMPT_TEMPLATE.format(article=indefinite_article(name), name=name)


@cache
def class_prompts() -> tuple[str, ...]:
    """Return the caption for every label id (index-aligned with the labels)."""
    return tuple(make_prompt(name) for name in load_class_names())
