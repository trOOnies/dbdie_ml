from __future__ import annotations
import os
from typing import TYPE_CHECKING
from copy import deepcopy
from PIL import Image
from dbdie_ml.cropper import Cropper
from dbdie_ml.movable_report import MovableReport
from dbdie_ml.utils import pls, filter_mulitype

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from dbdie_ml.classes import CropType

CropperAlignments = dict[str, list[Cropper]]


class CropperSwarm:
    """Chain of `Croppers` that can be run in sequence."""

    def __init__(self, croppers: list[Cropper | list[Cropper]]) -> None:
        assert all(
            isinstance(cpp, Cropper) or all(isinstance(cpp_i, Cropper) for cpp_i in cpp)
            for cpp in croppers
        )
        self.croppers = croppers
        self.cropper_alignments: list[CropperAlignments] = [
            self._group_cropper_list(cpp) for cpp in croppers
        ]
        self._croppers_flat = [
            cpp
            for cpa in self.cropper_alignments
            for cpp_list in cpa.values()
            for cpp in cpp_list
        ]
        self._movable_report = None

    def __len__(self) -> int:
        return len(self.croppers)

    def __repr__(self) -> str:
        """CropperSwarm(3 levels, 6 croppers)"""
        cps_levels = len(self)
        cps_croppers = len(self._croppers_flat)
        s = pls("level", cps_levels) + ", " + pls("cropper", cps_croppers)
        return f"CropperSwarm({s})"

    def print_croppers(self, verbose: bool = False) -> None:
        print("CROPPER SWARM:", str(self))
        print("CROPPERS:")
        if verbose:
            for cpp in self.croppers:
                print(f"- {cpp}")
        else:
            for cpp in self.croppers:
                if isinstance(cpp, list):
                    s = [f"Cropper('{cpp_i.settings.name}')" for cpp_i in cpp]
                    print(f"- [{', '.join(s)}]")
                else:
                    print(f"- Cropper('{cpp.settings.name}')")

    def get_all_fmts(self) -> list:
        return sum((cpp.full_model_types for cpp in self._croppers_flat), [])

    @property
    def cropper_flat_names(self) -> list:
        return [cpp.name for cpp in self._croppers_flat]

    # * Instantiate

    @classmethod
    def from_types(cls, ts: list["CropType" | list["CropType"]]) -> CropperSwarm:
        """Loads certain types of DBDIE Croppers"""
        assert all(
            isinstance(t, str) or all(isinstance(t_i, str) for t_i in t) for t in ts
        )

        ts_flattened = [(t if isinstance(t, list) else [t]) for t in ts]
        ts_flattened = sum(ts_flattened, [])
        assert len(ts_flattened) == len(set(ts_flattened))

        cppsw = CropperSwarm(
            [
                (
                    Cropper.from_type(t)
                    if isinstance(t, str)
                    else [Cropper.from_type(t_i) for t_i in t]
                )
                for t in ts
            ]
        )
        return cppsw

    # * Process Croppers

    @staticmethod
    def _group_cropper_list(croppers: Cropper | list[Cropper]) -> CropperAlignments:
        """Group a list of Croppers using the source folders they point to"""
        if isinstance(croppers, list):
            unique_srcs = set(cpp.settings.src for cpp in croppers)
            return {
                k: [cpp for cpp in croppers if cpp.settings.src == k]
                for k in unique_srcs
            }
        else:
            return {croppers.settings.src: [croppers]}

    # * Cropping

    @staticmethod
    def _apply_cropper(cpp: Cropper, img: "PILImage", src_filename: str) -> None:
        """Make all the `Cropper` crops for a single in-memory image,
        and save them in the settings 'dst' folder,
        inside the corresponding subfolder
        """
        # TODO: full_model_types?

        plain = src_filename[:-4]
        o = cpp.settings.offset

        for full_model_type in cpp.full_model_types:
            boxes = deepcopy(cpp.settings.crops[full_model_type])
            dst_fd = os.path.join(
                cpp.settings.dst,
                full_model_type,
            )
            for i, box in enumerate(boxes):
                cropped = img.crop(box)
                cropped.save(os.path.join(dst_fd, f"{plain}_{i+o}.jpg"))
                del cropped

    def _run_cropper(self, cpp: Cropper) -> None:
        """Run a single `Cropper`"""
        src = cpp.settings.src
        fs = self._movable_report.load_and_filter(src)
        for f in fs:
            img = Image.open(os.path.join(src, f))
            img = img.convert("RGB")
            self._apply_cropper(cpp, img, f)
            del img

    def _filter_use_croppers(self, use_croppers: str | list[str] | None) -> list[str]:
        possible_values = self.cropper_flat_names
        return filter_mulitype(
            use_croppers,
            default=possible_values,
            possible_values=possible_values,
        )

    def run_in_sequence(
        self, move: bool = True, use_croppers: list[str] | None = None
    ) -> None:
        """[OLD] Run all `Croppers` in their preset order"""
        cpp_to_use = self._filter_use_croppers(use_croppers)

        self._movable_report = MovableReport()
        for cpp in self.croppers:
            if isinstance(cpp, list):
                # TODO: Could be parallelized
                for cpp_i in cpp:
                    if cpp_i.name in cpp_to_use:
                        self._run_cropper(cpp_i)
            else:  # type: Cropper
                if cpp.name in cpp_to_use:
                    self._run_cropper(cpp)

        if move:
            self.move_images()
        self._movable_report = None

    def run(self, move: bool = True, use_croppers: list[str] | None = None) -> None:
        """[NEW] Run all `Croppers` iterating on images first"""
        # TODO: Implement 'use_crops'
        cpp_to_use = self._filter_use_croppers(use_croppers)

        self._movable_report = MovableReport()
        for cpa in self.cropper_alignments:
            # TODO: Different alignments but at-same-level could be parallelized
            for src, croppers in cpa.items():
                fs = self._movable_report.load_and_filter(src)
                for f in fs:
                    img = Image.open(os.path.join(src, f))
                    img = img.convert("RGB")
                    for cpp in croppers:
                        if cpp.name in cpp_to_use:
                            self._apply_cropper(cpp, img, f)
                    del img

        if move:
            self.move_images()
        self._movable_report = None

    # * Moving

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        self._movable_report.move_images()
