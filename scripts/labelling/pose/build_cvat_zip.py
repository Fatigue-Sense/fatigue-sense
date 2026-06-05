"""Pack pose labels into a CVAT-importable zip.

CVAT's Ultralytics YOLO Pose 1.0 importer expects ``data.yaml``, stub
images, and label txt files per split.

Usage::

    python -m scripts.labelling.pose.build_cvat_zip --split train
    python -m scripts.labelling.pose.build_cvat_zip --split val
    python -m scripts.labelling.pose.build_cvat_zip --split test
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from scripts.labelling.pose._defaults import DEFAULT_LABELS_ROOT, DEFAULT_OUTPUT_ROOT

DATA_YAML_TEMPLATE = """\
path: .
{split}: images/{split}
kpt_shape: [5, 3]
flip_idx: [0, 2, 1, 4, 3]
names:
  0: person
"""


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    split_dir = args.labels_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found")

    txt_files = sorted(split_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt labels in {split_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = args.out_dir / f"cvat_labels_{args.split}.zip"

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", DATA_YAML_TEMPLATE.format(split=args.split))
        for txt in txt_files:
            stem = txt.stem
            zf.write(txt, arcname=f"labels/{args.split}/{txt.name}")
            zf.writestr(f"images/{args.split}/{stem}.jpg", b"")

    print(
        f"Wrote {out_zip} with {len(txt_files)} labels "
        f"+ {len(txt_files)} image stubs + data.yaml"
    )


if __name__ == "__main__":
    main()
