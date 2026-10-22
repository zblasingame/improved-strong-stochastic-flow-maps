"""ImageReward, PickScore and HPSv2.1 scorers used by ``scripts/imagenet/score.py``.

Each builder returns ``score(uint8 NCHW images on device, labels) -> (N,)``,
scoring every image against its class caption ``"A photo of a/an {class name}"``.
All models are frozen, batched and run under bf16 autocast; the models' own
CLIP-style preprocessing (shorter-side bicubic resize to 224, center crop, CLIP
normalization) is reproduced on uint8 tensors. Imported only from the isolated
``scripts/imagenet/scoring_env`` environment.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


from src.evaluation.class_names import class_prompts  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MODEL_IDS = {
    'imagereward': 'THUDM/ImageReward ImageReward-v1.0',
    'pickscore': 'yuvalkirstain/PickScore_v1 (processor laion/CLIP-ViT-H-14-laion2B-s32B-b79K)',
    'hpsv2.1': 'xswu/HPSv2 HPS_v2.1_compressed.pt on open_clip ViT-H-14',
}
PROMPT_NOTE = 'A photo of a/an {class name}'
Scorer = Callable[[Tensor, np.ndarray], Tensor]  # (uint8 NCHW on device, labels) -> (N,)


def clip_preprocess(images: Tensor, size: int = 224) -> Tensor:
    """CLIP-style eval transform on uint8 NCHW: shorter-side bicubic resize, crop, normalize."""
    height, width = images.shape[-2:]
    scale = size / min(height, width)
    new_hw = (max(size, round(height * scale)), max(size, round(width * scale)))
    x = F.interpolate(images.float() / 255.0, size=new_hw, mode='bicubic', antialias=True)
    top, left = (new_hw[0] - size) // 2, (new_hw[1] - size) // 2
    x = x[..., top : top + size, left : left + size].clamp(0.0, 1.0)
    mean = x.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def frozen(model: torch.nn.Module) -> torch.nn.Module:
    """Eval mode with gradients disabled, so no activations are ever retained."""
    return model.eval().requires_grad_(False)


class PromptCache:
    """Encode each of the 1000 prompts once and gather per-sample rows by label."""

    @torch.inference_mode()
    def __init__(self, encode: Callable[[list[str]], Tensor], chunk: int = 100) -> None:
        prompts = list(class_prompts())
        self.table = torch.cat(
            [encode(prompts[i : i + chunk]) for i in range(0, len(prompts), chunk)]
        )

    def __call__(self, labels: np.ndarray) -> Tensor:
        """Return the cached text embedding of each sample's class prompt."""
        return self.table[torch.as_tensor(labels, device=self.table.device)]


def build_pickscore(device: torch.device) -> Scorer:
    """PickScore = logit_scale.exp() * cos(text, image) (official model-card formula)."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    model = frozen(AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').to(device))

    def encode_text(prompts: list[str]) -> Tensor:
        tokens = processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors='pt'
        ).to(device)
        return F.normalize(model.get_text_features(**tokens), dim=-1)

    texts = PromptCache(encode_text)
    scale = model.logit_scale.exp()

    def score(images: Tensor, labels: np.ndarray) -> Tensor:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            feats = model.get_image_features(pixel_values=clip_preprocess(images))
        return scale * (F.normalize(feats.float(), dim=-1) * texts(labels).float()).sum(-1)

    return score


def build_hpsv21(device: torch.device) -> Scorer:
    """HPSv2.1 = raw cosine(image, text) of the HPS-finetuned ViT-H-14 (no logit scale)."""
    import open_clip
    from huggingface_hub import hf_hub_download

    model = open_clip.create_model('ViT-H-14', pretrained=None, device=device)
    state = torch.load(hf_hub_download('xswu/HPSv2', 'HPS_v2.1_compressed.pt'), map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model = frozen(model)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    texts = PromptCache(lambda p: F.normalize(model.encode_text(tokenizer(p).to(device)), dim=-1))

    def score(images: Tensor, labels: np.ndarray) -> Tensor:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            feats = model.encode_image(clip_preprocess(images))
        return (F.normalize(feats.float(), dim=-1) * texts(labels).float()).sum(-1)

    return score


def build_imagereward(device: torch.device) -> Scorer:
    """Batched re-implementation of ``ImageReward.score`` (same modules and normalization)."""
    import ImageReward as RM

    model = frozen(RM.load('ImageReward-v1.0', device=device))
    prompts = list(class_prompts())
    tokens = model.blip.tokenizer(
        prompts, padding='max_length', truncation=True, max_length=35, return_tensors='pt'
    ).to(device)

    def score(images: Tensor, labels: np.ndarray) -> Tensor:
        index = torch.as_tensor(labels, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            embeds = model.blip.visual_encoder(clip_preprocess(images))
            atts = torch.ones(embeds.shape[:-1], dtype=torch.long, device=device)
            out = model.blip.text_encoder(
                tokens.input_ids[index],
                attention_mask=tokens.attention_mask[index],
                encoder_hidden_states=embeds,
                encoder_attention_mask=atts,
                return_dict=True,
            )
        rewards = model.mlp(out.last_hidden_state[:, 0, :].float()).squeeze(-1)
        return (rewards - model.mean) / model.std

    return score


BUILDERS = {
    'imagereward': build_imagereward,
    'pickscore': build_pickscore,
    'hpsv2.1': build_hpsv21,
}


def class_means(values: np.ndarray, labels: np.ndarray, num_classes: int = 1000) -> list[float]:
    """Mean score per class id (NaN for absent classes)."""
    sums = np.bincount(labels, weights=values, minlength=num_classes)
    counts = np.bincount(labels, minlength=num_classes)
    return [float(s / c) if c else float('nan') for s, c in zip(sums, counts, strict=True)]
