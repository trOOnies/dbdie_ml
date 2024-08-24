from __future__ import annotations
from typing import TYPE_CHECKING
from dbdie_ml.cropper import Cropper
from dbdie_ml.movable_report import MovableReport
from dbdie_ml.utils import pls
from dbdie_ml.code.cropper_swarm import (
    cropper_fmts_nand,
    filter_use_croppers,
    run_cropper,
    run_using_cropper_names,
    run_using_fmts,
)

if TYPE_CHECKING:
    from dbdie_ml.classes.base import RelPath, CropType, FullModelType


class CropperAlignments:
    """Group a list of Croppers into `CropperAlignments`.

    A `CropperAlignments` dict maps folders to many `Croppers`.
    In short, it's a way to take advantage of same-level `Croppers`
    that share a source folder, so that the amount of times an image is loaded
    is minimized.
    """

    def __init__(
        self,
        croppers: Cropper | list[Cropper],
    ):
        if isinstance(croppers, list):
            unique_srcs = set(cpp.settings.src_fd_rp for cpp in croppers)
            self._data: dict["RelPath", list[Cropper]] = {
                k: [cpp for cpp in croppers if cpp.settings.src_fd_rp == k]
                for k in unique_srcs
            }
        elif isinstance(croppers, Cropper):
            self._data: dict["RelPath", list[Cropper]] = {
                croppers.settings.src_fd_rp: [croppers]
            }
        else:
            raise TypeError

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __getitem__(self, key) -> list[Cropper]:
        return self._data[key]

    def show_mapping(self):
        return {k: [cp.name for cp in cp_list] for k, cp_list in self._data.items()}


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

    Nota: the `CropperSwarm` constructs in the background a list of `CropperAlignments`.
    See `CropperAlignments` for more information.
    """

    def __init__(self, croppers: list[Cropper | list[Cropper]]) -> None:
        assert all(
            isinstance(cpp, Cropper) or all(isinstance(cpp_i, Cropper) for cpp_i in cpp)
            for cpp in croppers
        ), "'croppers' list can only have Croppers and/or lists of Croppers."

        self.croppers = croppers
        self.cropper_alignments = [CropperAlignments(cpp) for cpp in croppers]

        self._croppers_flat = [
            cpp
            for cpa in self.cropper_alignments
            for cpp_list in cpa.values()
            for cpp in cpp_list
        ]
        self.version_range = self._croppers_flat[0].settings.version_range
        assert all(
            cpp.settings.version_range == self.version_range
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
        s = ", ".join(
            [
                f"'{self.version_range}'",
                pls("level", cps_levels),
                pls("cropper", cps_croppers),
            ]
        )
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
                    print(f"- [{', '.join(s)}]")
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

    def run_in_sequence(
        self,
        move: bool = True,
        use_croppers: list[str] | None = None,
    ) -> None:
        """[OLD] Run all `Croppers` in their preset order"""
        cpp_to_use = filter_use_croppers(self.cropper_flat_names, use_croppers)

        self._movable_report = MovableReport()
        for cpp in self.croppers:
            if isinstance(cpp, list):
                # TODO: Could be parallelized
                for cpp_i in cpp:
                    if cpp_i.name in cpp_to_use:
                        run_cropper(cpp_i, mr=self._movable_report)
            else:  # type: Cropper
                if cpp.name in cpp_to_use:
                    run_cropper(cpp, mr=self._movable_report)

        if move:
            self.move_images()
        self._movable_report = None

    # * Cropping (using CropperAlignments)

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
        cropper_fmts_nand(use_croppers, use_fmts)

        self._movable_report = MovableReport()

        if use_fmts is not None:
            run_using_fmts(self.cropper_alignments, self._movable_report, use_fmts)
        else:
            # This will use the full list of Croppers if use_croppers is None
            cpp_to_use = filter_use_croppers(self.cropper_flat_names, use_croppers)
            run_using_cropper_names(
                self.cropper_alignments, self._movable_report, cpp_to_use
            )

        if move:
            self.move_images()
        self._movable_report = None

    # * Moving

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        self._movable_report.move_images()
