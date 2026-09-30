"""Frozen query representations used by the paper profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def resolve_device(requested="auto"):
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


class MiniLMEncoder:
    def __init__(self, *, device="auto", cache_dir=None, batch_size=128):
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision=MINILM_REVISION,
            device=self.device,
            cache_folder=None if cache_dir is None else str(cache_dir),
        )

    def encode(self, texts):
        return np.asarray(
            self.model.encode(
                [str(text) for text in texts],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


class CLIPEncoder:
    def __init__(self, *, device="auto", batch_size=128):
        import clip

        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.clip = clip
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.model.eval()

    def encode(self, texts, images):
        import torch
        from PIL import Image

        text_parts = []
        image_parts = []
        texts = [str(text) for text in texts]
        images = list(images)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.clip.tokenize(
                    texts[start : start + self.batch_size], truncate=True
                ).to(self.device)
                values = self.model.encode_text(tokens).float()
                values /= values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                text_parts.append(values.cpu().numpy())
            for start in range(0, len(images), self.batch_size):
                batch = []
                for item in images[start : start + self.batch_size]:
                    try:
                        image = Image.open(str(item)).convert("RGB")
                        batch.append(self.preprocess(image))
                    except Exception as error:
                        raise RuntimeError(
                            f"Could not load MMR-Bench image {item!r}"
                        ) from error
                tensor = torch.stack(batch).to(self.device).float()
                values = self.model.encode_image(tensor).float()
                values /= values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                image_parts.append(values.cpu().numpy())
        combined = np.concatenate(
            [np.concatenate(text_parts), np.concatenate(image_parts)], axis=1
        ).astype(np.float32)
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        return combined


def cached_embeddings(path, build):
    destination = Path(path)
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            return archive["train"], archive["test"]
    train, test = build()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, train=train, test=test)
    return train, test
