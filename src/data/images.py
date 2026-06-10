"""Image representations: the signal rendered as a clinical ECG plot (rgb) or
stacked into a 3-channel matrix (stacked_signal), then run through the vision
encoder's own processor. Only pixel features are produced — the vision text
branch is never used."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from registry import ENCODERS

GRID_MAJOR, GRID_MINOR = (0.85, 0.25, 0.25, 0.45), (0.90, 0.70, 0.70, 0.35)
SIGNAL_COLOR, BACKGROUND = (0.0, 0.0, 0.55), (1.0, 0.98, 0.96)


def plot_ecg(ecg, sample_rate=250, row_height=2.5):
    """(leads, samples) -> uint8 RGB array of a standard-grid ECG plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    leads, samples = ecg.shape
    secs = samples / sample_rate
    fig, ax = plt.subplots(figsize=(secs * 2.0, leads * row_height * 1.1), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, secs)
    ax.set_ylim(-(leads - 1) * row_height - row_height * 0.8, row_height * 0.8)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(which="major", linewidth=0.4, color=GRID_MAJOR)
    ax.grid(which="minor", linewidth=0.2, color=GRID_MINOR)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    t = np.linspace(0, secs, samples, endpoint=False)
    for i in range(leads):
        ax.plot(t, ecg[i] - i * row_height, linewidth=0.6, color=SIGNAL_COLOR)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


class _VisionRep:
    def __init__(self):
        self.processor = None
        self.augment = None

    def _features(self, ds, image):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(ENCODERS[ds.cfg.encoder].hf_processor, use_fast=True)
            if ds.cfg.augment_rgb and ds.cfg.mode == "train":
                from torchvision import transforms as T
                self.augment = T.Compose([
                    T.RandomApply([T.ColorJitter(brightness=0.2)], p=0.5),
                    T.RandomApply([T.RandomRotation(5)], p=0.5),
                    T.RandomApply([T.GaussianBlur(5, sigma=(0.0, 1.5))], p=0.5),
                    T.RandomApply([T.ColorJitter(hue=0.08, saturation=0.3)], p=0.5),
                ])
        if self.augment is not None:
            image = self.augment(image)
        out = self.processor(images=image, return_tensors="pt")
        return {k: v[0] if isinstance(v, torch.Tensor) else torch.as_tensor(v[0]) for k, v in out.items()}

    def splice(self, ds, row, ids, labels):
        return ids, labels


class RGB(_VisionRep):
    def features(self, ds, row):
        return self._features(ds, Image.fromarray(plot_ecg(ds.load_signal(row))))


class StackedSignal(_VisionRep):
    def features(self, ds, row):
        signal = ds.load_signal(row)
        stacked = np.stack([signal * 255] * 3, axis=-1).astype(np.uint8)
        return self._features(ds, Image.fromarray(stacked))
