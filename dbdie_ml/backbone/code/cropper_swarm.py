import os
from copy import deepcopy
from typing import TYPE_CHECKING, Union

from dbdie_classes.paths import absp
from dbdie_classes.utils import filter_multitype
from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from dbdie_classes.base import CropType, Filename, FullModelType
    from backbone.cropping.cropper import Cropper
    from backbone.cropping.movable_report import MovableReport

# * Instantiate


def flatten_cpas(cpas):
    """Flatten CropperAligments into a Cropper list."""
    return [
        cpp
        for cpa in cpas
        for cpp_list in cpa.values()
        for cpp in cpp_list
    ]


def check_croppers_dbdvr(croppers_flat, version_range) -> None:
    assert all(
        cpp.settings.version_range == version_range
        for cpp in croppers_flat
    ), "All croppers version ranges must exactly coincide"


def check_cropper_types(ts: list[Union["CropType", list["CropType"]]]) -> None:
    assert all(
        isinstance(t, str) or all(isinstance(t_i, str) for t_i in t) for t in ts
    )

    ts_flattened = [(t if isinstance(t, list) else [t]) for t in ts]
    ts_flattened = sum(ts_flattened, [])
    assert len(ts_flattened) == len(set(ts_flattened))


# * Cropping helpers


def filter_use_croppers(
    cropper_flat_names,
    use_croppers: str | list[str] | None,
) -> list[str]:
    return filter_multitype(
        use_croppers,
        default=cropper_flat_names,
        possible_values=cropper_flat_names,
    )


def apply_cropper(
    cpp: "Cropper",
    img: "PILImage",
    src_filename: "Filename",
    full_model_types: str | list[str] | None = None,
) -> None:
    """Make all the `Cropper` crops for a single in-memory image,
    and save them in the settings 'dst' folder,
    inside the corresponding subfolder
    """
    fmts = cpp._filter_fmts(full_model_types)

    plain = src_filename[:-4]
    o = cpp.settings.offset if cpp.settings.offset is not None else 0

    for fmt in fmts:
        boxes = deepcopy(cpp.settings.crops[fmt])
        dst_fd = os.path.join(cpp.settings.dst, fmt)
        for i, box in enumerate(boxes):
            cropped = img.crop(box.raw())
            cropped.save(os.path.join(dst_fd, f"{plain}_{i+o}.jpg"))
            del cropped


# * Cropping (in sequence)


def run_cropper(cpp: "Cropper", mr: "MovableReport") -> None:
    """Run a single `Cropper`"""
    src = cpp.settings.src
    fs = mr.load_and_filter(src)
    for f in fs:
        img = Image.open(os.path.join(src, f))
        img = img.convert("RGB")
        apply_cropper(cpp, img, f)
        del img


# * Cropping (using CropperAlignments)


def cropper_fmts_nand(
    use_croppers: list[str] | None,
    use_fmts: list["FullModelType"] | None,
) -> None:
    c_none = use_croppers is None
    f_none = use_fmts is None

    cond = c_none or f_none
    assert cond, "'use_croppers' and 'use_fmts' cannot be used at the same time"

    if not c_none:
        assert isinstance(use_croppers, list) and use_croppers
    elif not f_none:
        assert isinstance(use_fmts, list) and use_fmts


def run_using_fmts(
    cpas,  # list["CropperAlignments"]
    mr: "MovableReport",
    use_fmts: list["FullModelType"],
) -> None:
    """Run filtering on `FullModelTypes`"""
    for cpa in cpas:
        print("-", cpa.show_mapping())
        # TODO: Different alignments but at-same-level could be parallelized
        for src_rp, croppers in cpa.items():
            src = absp(src_rp)
            fs = mr.load_and_filter(src)
            for f in fs:
                img = Image.open(os.path.join(src, f))
                img = img.convert("RGB")
                for cpp in croppers:
                    found_fmts = [
                        fmt for fmt in use_fmts if fmt in cpp.full_model_types_set
                    ]
                    if found_fmts:
                        apply_cropper(
                            cpp,
                            img,
                            src_filename=f,
                            full_model_types=found_fmts,
                        )
                del img


def run_using_cropper_names(
    cpas,  # list["CropperAlignments"]
    mr: "MovableReport",
    use_croppers: list[str],
) -> None:
    """Run filtering on `Cropper` names"""
    for cpa in cpas:
        print("-", cpa.show_mapping())
        # TODO: Different alignments but at-same-level could be parallelized
        for src_rp, croppers in cpa.items():
            src = absp(src_rp)
            fs = mr.load_and_filter(src)
            for f in fs:
                img = Image.open(os.path.join(src, f))
                img = img.convert("RGB")
                for cpp in croppers:
                    if cpp.name in use_croppers:
                        apply_cropper(cpp, img, src_filename=f)
                del img
