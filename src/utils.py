import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class WristXrayDataset(Dataset):
    """GRAZPEDWRI-DX wrist X-rays from a Roboflow COCO export.

    Expects a split directory containing the images and `_annotations.coco.json`
    (e.g. data/GRAZPEDWRI-DX.v5i.coco/train).

    Roboflow adds a placeholder category (supercategory "none", id 0) that no
    annotation uses; it is dropped. The remaining category ids are remapped to
    contiguous indices 0..C-1, listed in `self.category_names`.
    `self.classes` / `self.num_classes` always describe what the model outputs.

    task="multiclass" (single-label, for nn.CrossEntropyLoss): every image
        gets exactly one class out of
            classes = [negative_class, *target_classes]
        target_classes is a priority list: an image is labelled with the first
        target class it has a box for, otherwise negative_class. The default
        target_classes=("fracture",) gives binary ["no_fracture", "fracture"].
        Returns (image, one_hot) with one_hot float32 [num_classes] (so
        `labels.argmax(1)` and `labels.shape[1]` work), or a scalar int64 class
        index with one_hot=False.
    task="classification" (multi-label, for nn.BCEWithLogitsLoss): returns
        (image, multi_hot) where multi_hot is float32 [C], 1 if the image has at
        least one box of that class.
    task="detection": returns (image, target) where target is a torchvision
        style dict: boxes (xyxy, float32, as tv_tensors.BoundingBoxes so
        torchvision.transforms.v2 updates them), labels (int64, 1..C; 0 is
        reserved for background), image_id, area, iscrowd.

    `exclude_classes` drops those class names entirely (e.g. "text" for the
    burned-in laterality markers).
    """

    ANN_FILE = "_annotations.coco.json"

    TASKS = ("multiclass", "classification", "detection")

    def __init__(self, split_dir, transform=None, task="multiclass",
                 target_classes=("fracture",), negative_class=None, one_hot=True,
                 exclude_classes=(), ann_file=None):
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got {task!r}")

        self.image_dir = Path(split_dir)
        self.transform = transform
        self.task = task
        self.one_hot = one_hot

        ann_path = Path(ann_file) if ann_file else self.image_dir / self.ANN_FILE
        with open(ann_path) as f:
            coco = json.load(f)

        exclude = set(exclude_classes)
        used_cat_ids = {a["category_id"] for a in coco["annotations"]}
        cats = sorted(
            (c for c in coco["categories"]
             if not (c.get("supercategory") == "none" and c["id"] not in used_cat_ids)
             and c["name"] not in exclude),
            key=lambda c: c["id"],
        )
        self.category_names = [c["name"] for c in cats]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

        if task == "multiclass":
            target_classes = list(target_classes)
            missing = [c for c in target_classes if c not in self.category_names]
            if missing:
                raise ValueError(f"target_classes {missing} not in {self.category_names}")
            if not target_classes or len(set(target_classes)) != len(target_classes):
                raise ValueError("target_classes must be a non-empty list without duplicates")
            if negative_class is None:
                negative_class = (f"no_{target_classes[0]}" if len(target_classes) == 1
                                  else "other")
            self.classes = [negative_class, *target_classes]
            # category index -> output class, highest priority first
            self._priority = [(self.category_names.index(c), i + 1)
                              for i, c in enumerate(target_classes)]
        else:
            self.classes = list(self.category_names)

        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self.anns_by_image = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a["category_id"] in self.cat_id_to_idx and a["image_id"] in self.anns_by_image:
                self.anns_by_image[a["image_id"]].append(a)

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.images)

    def _load_image(self, info):
        with Image.open(self.image_dir / info["file_name"]) as img:
            return img.convert("RGB")

    def _multi_hot(self, anns):
        label = torch.zeros(len(self.category_names), dtype=torch.float32)
        for a in anns:
            label[self.cat_id_to_idx[a["category_id"]]] = 1.0
        return label

    def _class_index(self, anns):
        present = {self.cat_id_to_idx[a["category_id"]] for a in anns}
        for cat_idx, out_idx in self._priority:
            if cat_idx in present:
                return out_idx
        return 0

    def _single_label(self, anns):
        idx = self._class_index(anns)
        if not self.one_hot:
            return torch.tensor(idx, dtype=torch.int64)
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        label[idx] = 1.0
        return label

    @property
    def targets(self):
        """Class index per image (multiclass only), e.g. for class weights or
        a WeightedRandomSampler, without decoding any images."""
        if self.task != "multiclass":
            raise AttributeError("targets is only defined for task='multiclass'")
        return [self._class_index(self.anns_by_image[im["id"]]) for im in self.images]

    def _detection_target(self, info, anns, image_size):
        from torchvision import tv_tensors

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[a["category_id"]] + 1)
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        h, w = image_size
        return {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format="XYXY", canvas_size=(h, w)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": info["id"],
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

    def __getitem__(self, idx):
        info = self.images[idx]
        anns = self.anns_by_image[info["id"]]
        image = self._load_image(info)

        if self.task != "detection":
            if self.transform:
                image = self.transform(image)
            label = self._single_label(anns) if self.task == "multiclass" else self._multi_hot(anns)
            return image, label

        target = self._detection_target(info, anns, (image.height, image.width))
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    @staticmethod
    def detection_collate(batch):
        """collate_fn for detection DataLoaders (variable number of boxes)."""
        return tuple(zip(*batch))


DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "GRAZPEDWRI-DX.v5i.coco"
SPLITS = ("train", "valid", "test")


def main(data_root=DATA_ROOT, batch_size=32, num_workers=0, max_batches=None):
    """Smoke test: build every split in both modes and iterate a DataLoader."""
    import time
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    cls_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    det_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    for split in SPLITS:
        split_dir = Path(data_root) / split
        print(f"\n=== {split} ({split_dir}) ===")

        # Image-level labels, default collate.
        # multiclass: one-hot, exactly one class per image (CrossEntropyLoss).
        # classification: multi-hot over all categories (BCEWithLogitsLoss).
        for task in ("multiclass", "classification"):
            ds = WristXrayDataset(split_dir, task=task, transform=cls_tf)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                num_workers=num_workers)
            counts = torch.zeros(ds.num_classes)
            n_imgs, t0 = 0, time.time()
            for i, (images, labels) in enumerate(loader):
                assert images.shape[1:] == (3, 224, 224), images.shape
                assert labels.shape == (images.shape[0], ds.num_classes), labels.shape
                if task == "multiclass":
                    assert (labels.sum(1) == 1).all(), "multiclass label must be one-hot"
                counts += labels.sum(0)
                n_imgs += images.shape[0]
                if max_batches and i + 1 >= max_batches:
                    break
            print(f"[{task:<14s}] {n_imgs}/{len(ds)} images in {time.time() - t0:.1f}s, "
                  f"last batch {tuple(images.shape)}")
            print("  images per class:",
                  {c: int(n) for c, n in zip(ds.classes, counts)})

        # Detection: variable number of boxes per image, custom collate.
        ds = WristXrayDataset(split_dir, task="detection", exclude_classes=["text"],
                              transform=det_tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            collate_fn=WristXrayDataset.detection_collate)
        n_imgs = n_boxes = n_empty = 0
        t0 = time.time()
        for i, (images, targets) in enumerate(loader):
            for img, tgt in zip(images, targets):
                boxes = tgt["boxes"]
                assert boxes.ndim == 2 and boxes.shape[1] == 4
                assert len(boxes) == len(tgt["labels"])
                if len(boxes):
                    h, w = img.shape[-2:]
                    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
                    assert boxes.min() >= 0 and (boxes[:, [0, 2]] <= w).all() and (boxes[:, [1, 3]] <= h).all()
                    assert tgt["labels"].min() >= 1 and tgt["labels"].max() <= ds.num_classes
                n_boxes += len(boxes)
                n_empty += len(boxes) == 0
            n_imgs += len(images)
            if max_batches and i + 1 >= max_batches:
                break
        print(f"[detection]      {n_imgs}/{len(ds)} images in {time.time() - t0:.1f}s, "
              f"{n_boxes} boxes, {n_empty} images without boxes (text excluded)")

    print("\nAll splits loaded OK.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smoke-test WristXrayDataset on all splits.")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="worker processes (startup is slow on macOS spawn; 0 is fastest here)")
    p.add_argument("--max-batches", type=int, default=None,
                   help="stop each loader after N batches (default: full pass)")
    a = p.parse_args()
    main(a.data_root, a.batch_size, a.num_workers, a.max_batches)
