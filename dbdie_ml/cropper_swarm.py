from __future__ import annotations
import os
from typing import TYPE_CHECKING
from copy import deepcopy
from PIL import Image
from dbdie_ml.cropper import Cropper
from dbdie_ml.movable_report import MovableReport
if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from dbdie_ml.classes import CropType

CropperAlignments = dict[str, list[Cropper]]


class CropperSwarm:
    """Chain of `Croppers` that can be run in sequence."""
    def __init__(
        self,
        croppers: list[Cropper | list[Cropper]]
    ) -> None:
        assert all(
            isinstance(cpp, Cropper) or all(isinstance(cpp_i, Cropper) for cpp_i in cpp)
            for cpp in croppers
        )
        self.croppers = croppers
        self.cropper_alignments: list[CropperAlignments] = [
            self._group_cropper_list(cpp)
            for cpp in croppers
        ]
        self._movable_report = None

    # * Instantiate

    def print_croppers(self) -> None:
        print("CROPPER SWARM:", str(self))
        print("CROPPERS:")
        for cpp in self.croppers:
            print(f"- {cpp}")

    @classmethod
    def from_types(
        cls,
        ts: list["CropType" | list["CropType"]]
    ) -> CropperSwarm:
        """Loads certain types of DBDIE Croppers"""
        assert all(
            isinstance(t, str) or all(isinstance(t_i, str) for t_i in t)
            for t in ts
        )

        ts_flattened = [
            (t if isinstance(t, list) else [t])
            for t in ts
        ]
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
    def _group_cropper_list(
        croppers: Cropper | list[Cropper]
    ) -> CropperAlignments:
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

    def _apply_cropper(
        self,
        cpp: Cropper,
        img: "PILImage",
        src_filename: str
    ) -> None:
        """Make all the `Cropper` crops for a single in-memory image,
        and save them in the settings 'dst' folder,
        inside the corresponding subfolder
        """
        plain = src_filename[:-4]
        o = cpp.settings.offset

        for full_model_type in self.full_model_types:
            boxes = deepcopy(self.settings.crops[full_model_type])
            dst_fd = os.path.join(
                self.settings.dst,
                full_model_type,
            )
            for i, box in enumerate(boxes):
                cropped = img.crop(box)
                cropped.save(
                    os.path.join(dst_fd, f"{plain}_{i+o}.jpg")
                )
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

    def run_in_sequence(self) -> None:
        """[OLD] Run all `Croppers` in their preset order"""
        self._movable_report = MovableReport()
        for cpp in self.croppers:
            if isinstance(cpp, list):
                # TODO: Could be parallelized
                for cpp_i in cpp:
                    self._run_cropper(cpp_i)
            else:  # type: Cropper
                self._run_cropper(cpp)

        self.move_images()
        self._movable_report = None

    def run(self) -> None:
        """[NEW] Run all `Croppers` iterating on images first"""
        # TODO: Implement again the 'crop_only' parameter
        self._movable_report = MovableReport()
        for cpa in self.cropper_alignments:
            # TODO: Different alignments but at-same-level could be parallelized
            for src, croppers in cpa.items():
                fs = self._movable_report.load_and_filter(src)
                for f in fs:
                    img = Image.open(os.path.join(src, f))
                    img = img.convert("RGB")
                    for cpp in croppers:
                        self._apply_cropper(cpp, img, f)
                    del img

        self.move_images()
        self._movable_report = None

    # * Moving

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        self._movable_report.move_images()
