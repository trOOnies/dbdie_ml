from __future__ import annotations
import os
from typing import TYPE_CHECKING
from copy import deepcopy
from PIL import Image
from dbdie_ml.paths import absp
from dbdie_ml.cropper import Cropper
from dbdie_ml.movable_report import MovableReport
from dbdie_ml.utils import pls, filter_multitype

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from dbdie_ml.classes import Filename, RelPath, CropType, FullModelType

CropperAlignments = dict["RelPath", list[Cropper]]


class CropperSwarm:
    """Chain of `Croppers` that can be run in sequence.
    This input sequence will be respected, so it's best to design
    it based on crop dependencies.

    Instantiation:
    >>> cps = CropperSwarm.from_types(
            [
                ["surv", "killer"],
                "surv_player",
                "killer_player",
            ]
        )
    or
    >>> cps = CropperSwarm(
        [
            [Cropper.from_type("surv"), Cropper.from_type("killer")],
            Cropper.from_type("surv_player"),
            Cropper.from_type("killer_player"),
        ]
    )

    Usage:
    >>> cps.print_croppers()
    >>> cps.run()  # ! doesn't return output, processes folder images in the background

    Nota: the `CropperSwarm` constructs in the background
    a list of `CropperAlignments`. See `_make_cropper_alignments()` for more information.
    """

    def __init__(self, croppers: list[Cropper | list[Cropper]]) -> None:
        assert all(
            isinstance(cpp, Cropper) or all(isinstance(cpp_i, Cropper) for cpp_i in cpp)
            for cpp in croppers
        ), "'croppers' list can only have Croppers and/or lists of Croppers."

        self.croppers = croppers
        self.cropper_alignments: list[CropperAlignments] = [
            self._make_cropper_alignments(cpp) for cpp in croppers
        ]

        self._croppers_flat = [
            cpp
            for cpa in self.cropper_alignments
            for cpp_list in cpa.values()
            for cpp in cpp_list
        ]
        self.version_range = self._croppers_flat[0].settings.version_range
        assert all(
            cpp.settings.version_range == self.version
            for cpp in self._croppers_flat
        ), "All croppers version ranges must exactly coincide"
        self.cropper_flat_names = [cpp.name for cpp in self._croppers_flat]

        self._movable_report = None

    # * Dunders and presentation

    def __len__(self) -> int:
        return len(self.croppers)

    def __repr__(self) -> str:
        """CropperSwarm('>=7.5.0', 3 levels, 6 croppers)"""
        cps_levels = len(self)
        cps_croppers = len(self._croppers_flat)
        s = ", ".join([
            f"'{self.version_range}'",
            pls("level", cps_levels),
            pls("cropper", cps_croppers),
        ])
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
                    s = [f"'{cpp_i.settings.name}'" for cpp_i in cpp]
                    print(f"- {s}")
                else:
                    print(f"- '{cpp.settings.name}'")

    def get_all_fmts(self) -> list:
        """Get all `FullModelTypes` present in its `Croppers`"""
        return sum((cpp.full_model_types for cpp in self._croppers_flat), [])

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

    @staticmethod
    def _make_cropper_alignments(
        croppers: Cropper | list[Cropper],
    ) -> CropperAlignments:
        """Group a list of Croppers into `CropperAlignments`.

        A `CropperAlignments` dict maps folders to many `Croppers`.
        In short, it's a way to take advantage of same-level `Croppers`
        that share a source folder, so that the amount of times an image is loaded
        is minimized.
        """
        if isinstance(croppers, list):
            unique_srcs = set(cpp.settings.src_fd_rp for cpp in croppers)
            return {
                k: [cpp for cpp in croppers if cpp.settings.src_fd_rp == k]
                for k in unique_srcs
            }
        else:
            return {croppers.settings.src_fd_rp: [croppers]}

    # * Cropping helpers

    def _filter_use_croppers(self, use_croppers: str | list[str] | None) -> list[str]:
        possible_values = self.cropper_flat_names
        return filter_multitype(
            use_croppers,
            default=possible_values,
            possible_values=possible_values,
        )

    @staticmethod
    def _apply_cropper(
        cpp: Cropper,
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
        o = cpp.settings.offset

        for fmt in fmts:
            boxes = deepcopy(cpp.settings.crops[fmt])
            dst_fd = os.path.join(cpp.settings.dst, fmt)
            for i, box in enumerate(boxes):
                cropped = img.crop(box)
                cropped.save(os.path.join(dst_fd, f"{plain}_{i+o}.jpg"))
                del cropped

    # * Cropping (in sequence)

    def _run_cropper(self, cpp: Cropper) -> None:
        """Run a single `Cropper`"""
        src = cpp.settings.src
        fs = self._movable_report.load_and_filter(src)
        for f in fs:
            img = Image.open(os.path.join(src, f))
            img = img.convert("RGB")
            self._apply_cropper(cpp, img, f)
            del img

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

    # * Cropping (using CropperAlignments)

    def _cropper_fmts_nand(
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

    def _run_using_fmts(self, use_fmts: list["FullModelType"]) -> None:
        """Run filtering on `FullModelTypes`"""
        for cpa in self.cropper_alignments:
            # TODO: Different alignments but at-same-level could be parallelized
            for src_rp, croppers in cpa.items():
                src = absp(src_rp)
                fs = self._movable_report.load_and_filter(src)
                for f in fs:
                    img = Image.open(os.path.join(src, f))
                    img = img.convert("RGB")
                    for cpp in croppers:
                        found_fmts = [
                            fmt for fmt in use_fmts if fmt in cpp.full_model_types_set
                        ]
                        if found_fmts:
                            self._apply_cropper(
                                cpp,
                                img,
                                src_filename=f,
                                full_model_types=found_fmts,
                            )
                    del img

    def _run_using_cropper_names(self, use_croppers: list[str]) -> None:
        """Run filtering on `Cropper` names"""
        for cpa in self.cropper_alignments:
            # TODO: Different alignments but at-same-level could be parallelized
            for src_rp, croppers in cpa.items():
                src = absp(src_rp)
                fs = self._movable_report.load_and_filter(src)
                for f in fs:
                    img = Image.open(os.path.join(src, f))
                    img = img.convert("RGB")
                    for cpp in croppers:
                        if cpp.name in use_croppers:
                            self._apply_cropper(cpp, img, src_filename=f)
                    del img

    def run(
        self,
        move: bool = True,
        use_croppers: list[str] | None = None,
        use_fmts: list["FullModelType"] | None = None,
    ) -> None:
        """[NEW] Run all `Croppers` iterating on images first.

        move: Whether to move the source images at the end of the cropping.
            Note: The `MovableReport` still avoid creating crops
            of duplicate source images.

        Filter options (cannot be used at the same time):
        - use_croppers: Filter cropping using `Cropper` names (level=`Cropper`).
        - use_fmt: Filter cropping using `FullModelTypes` names (level=crop type).
        """
        self._cropper_fmts_nand(use_croppers, use_fmts)

        self._movable_report = MovableReport()

        if use_fmts is not None:
            self._run_using_fmts(use_fmts)
        else:
            # This will use the full list of Croppers if use_croppers is None
            cpp_to_use = self._filter_use_croppers(use_croppers)
            self._run_using_cropper_names(cpp_to_use)

        if move:
            self.move_images()
        self._movable_report = None

    # * Moving

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        self._movable_report.move_images()
