"""Download the two datasets and extract frozen Swin-B (ImageNet-22K) features.

Writes, for each of tiny_train, tiny_valid, cifar100_train, cifar100_test:
  features/<dataset>_<split>.npy          (n x 1024 float32, the input of the model's own classifier)
  features/<dataset>_<split>_labels.npy   (n integer labels)
and the 22K classifier head features/head_22k.npz (W, b), used by zero_shot_22k.py.
A split whose feature file exists is skipped. Runs on MPS if available (about an hour on an M3), else CPU.

Transforms follow LA-LoRA's DP_lora.py at commit 578e492, L135-148 (Tiny: Resize((224,224));
CIFAR-100: Resize(256) + CenterCrop(224); ImageNet mean/std).

Usage: .venv/bin/python -I our_src/extract_features.py <root>"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import timm
import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms as T

from common import DATASET_NAMES, EVAL_SPLIT, NUM_CLASSES, TINY_HF_ID, cifar_dir, feature_path, hf_cache_dir

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRANSFORM = {"tiny": T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)]),
             "cifar100": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])}
SIZES = {("tiny", "train"): 100000, ("tiny", "valid"): 10000, ("cifar100", "train"): 50000, ("cifar100", "test"): 10000}


class Images(torch.utils.data.Dataset):
    """Transformed images of one split, and their labels."""

    def __init__(self, root, dataset, split):
        self.dataset, self.transform = dataset, TRANSFORM[dataset]
        if dataset == "tiny":
            self.data = load_dataset(TINY_HF_ID, cache_dir=hf_cache_dir(root), split=split)
            self.labels = np.array(self.data["label"])
        else:
            self.data = torchvision.datasets.CIFAR100(cifar_dir(root), train=(split == "train"), download=True)
            self.labels = np.array(self.data.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]["image"] if self.dataset == "tiny" else self.data[i][0]
        return self.transform(image.convert("RGB"))


@torch.inference_mode()
def extract(model, images, device, batch_size=64, workers=4):
    """Pooled features of all images, in order: the 1024-dim vector the model's classifier head takes as input."""
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=workers, shuffle=False)
    return np.concatenate([model.forward_head(model.forward_features(x.to(device)), pre_logits=True).float().cpu().numpy()
                           for x in loader])


def main(root):
    os.makedirs(feature_path(root, ""), exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(MODEL_ID, pretrained=True).eval().to(device)
    head = model.head.fc
    np.savez(feature_path(root, "head_22k.npz"), W=head.weight.detach().cpu().numpy(), b=head.bias.detach().cpu().numpy())

    for dataset in DATASET_NAMES:
        for split in ("train", EVAL_SPLIT[dataset]):
            stem = "%s_%s" % (dataset, split)
            if os.path.exists(feature_path(root, stem + ".npy")):
                continue
            images = Images(root, dataset, split)
            n = SIZES[(dataset, split)]
            assert len(images) == n and (np.bincount(images.labels, minlength=NUM_CLASSES[dataset]) == n // NUM_CLASSES[dataset]).all()
            F = extract(model, images, device)
            assert F.shape == (n, 1024) and np.isfinite(F).all()
            np.save(feature_path(root, stem + "_labels.npy"), images.labels)
            np.save(feature_path(root, stem + ".npy"), F)
            print(stem, F.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
