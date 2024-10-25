from __future__ import annotations

from copy import deepcopy
from dbdie_classes.code.version import filter_images_with_dbdv
from dbdie_classes.options.CROP_TYPES import DEFAULT_CROP_TYPES_SEQ
from dbdie_classes.utils import pls
from typing import TYPE_CHECKING
import yaml

from backbone.classes.register import get_cropper_swarm_mpath
from backbone.code.cropper_swarm import (
    check_croppers_dbdvr,
    check_cropper_types,
    cropper_fmts_nand,
    filter_use_croppers,
    flatten_cpas,
    run_cropper,
    run_using_cropper_names,
    run_using_fmts,
)
from backbone.cropping.cropper import Cropper
from backbone.cropping.movable_report import MovableReport
from backbone.options.COLORS import get_class_cprint

if TYPE_CHECKING:
    from dbdie_classes.base import Filename, FullModelType, RelPath

csw_print = get_class_cprint("CropperSwarm")


class CropperAlignments:
    """Group a list of Croppers into CropperAlignments.

    A CropperAlignments dict maps folders to many Croppers.
    In short, it's a way to take advantage of same-level Croppers
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

    def items(self):
        return self._data.items()

    def show_mapping(self):
        return {k: [cp.name for cp in cp_list] for k, cp_list in self._data.items()}


class CropperSwarm:
    """Chain of Croppers that can be run in sequence.
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

    Nota: the CropperSwarm constructs in the background a list of CropperAlignments.
    See CropperAlignments for more information.
    """

    def __init__(
        self,
        id: int,
        name: str,
        dbdv_min_id: int,
        dbdv_max_id: int | None,
        croppers: list[Cropper | list[Cropper]],
    ) -> None:
        assert all(
            isinstance(cpp, Cropper) or all(isinstance(cpp_i, Cropper) for cpp_i in cpp)
            for cpp in croppers
        ), "'croppers' list can only have Croppers and/or lists of Croppers."

        self.id = id
        self.name = name
        self.dbdv_min_id = dbdv_min_id
        self.dbdv_max_id = dbdv_max_id

        self.croppers = croppers
        self.cropper_alignments = [CropperAlignments(cpp) for cpp in croppers]

        self._croppers_flat = flatten_cpas(self.cropper_alignments)
        self.dbdvr = self._croppers_flat[0].settings.dbdvr
        self.dbdvr_ids = [self.dbdv_min_id, self.dbdv_max_id]
        check_croppers_dbdvr(self._croppers_flat, self.dbdvr)
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
                f"'{self.dbdvr}'",
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

    def get_all_fmts(self) -> list["FullModelType"]:
        """Get all FullModelTypes present in its Croppers."""
        return sum((cpp.fmts for cpp in self._croppers_flat), [])

    def get_cs_that_contains_fmt(self, fmt: "FullModelType"):
        for cpp in self._croppers_flat:
            cs = cpp.settings
            if fmt in cs.crops:
                break
        else:
            raise AssertionError(f"No Cropper in the CropperSwarm contains the fmt '{fmt}'.")
        return cs

    # * Instantiate

    @classmethod
    def load_metadata(cls, name: str) -> dict:
        path = get_cropper_swarm_mpath(name)
        with open(path) as f:
            metadata = yaml.safe_load(f)
        assert metadata["name"] == name
        return metadata

    @classmethod
    def from_register(cls, name: str) -> CropperSwarm:
        """Loads a registered `CropperSwarm` with name `name`."""
        ts = deepcopy(DEFAULT_CROP_TYPES_SEQ)
        check_cropper_types(ts)

        metadata = cls.load_metadata(name)

        return CropperSwarm(
            id=metadata["id"],
            name=name,
            dbdv_min_id=metadata["dbdv_min_id"],
            dbdv_max_id=metadata["dbdv_max_id"],
            croppers=[
                (
                    Cropper.from_register(name, t)
                    if isinstance(t, str)
                    else [Cropper.from_register(name, t_i) for t_i in t]
                )
                for t in ts
            ],
        )

    def run_in_sequence(
        self,
        move: bool = True,
        use_croppers: list[str] | None = None,
    ) -> None:
        """[OLD] Run all Croppers in their preset order."""
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
        fs: list["Filename"],
        move: bool = True,
        use_croppers: list[str] | None = None,
        use_fmts: list["FullModelType"] | None = None,
    ) -> None:
        """[NEW] Run all Croppers iterating on images first.

        move: Whether to move the source images at the end of the cropping.
            Note: The MovableReport still avoid creating crops
            of duplicate source images.

        Filter options (cannot be used at the same time):
        - use_croppers: Filter cropping using Cropper names (level=Cropper).
        - use_fmt: Filter cropping using FullModelTypes names (level=crop type).
        """
        cropper_fmts_nand(use_croppers, use_fmts)

        self._movable_report = MovableReport(fs)

        csw_print("Generating crops...")
        if use_fmts is not None:
            run_using_fmts(self.cropper_alignments, self._movable_report, use_fmts)
        else:
            # This will use the full list of Croppers if use_croppers is None
            cpp_to_use = filter_use_croppers(self.cropper_flat_names, use_croppers)
            run_using_cropper_names(
                self.cropper_alignments,
                self._movable_report,
                cpp_to_use,
            )
        csw_print("Crops generated.")

        if move:
            self.move_images()
        self._movable_report = None

    # * Moving

    def filter_fs_with_dbdv(self, matches: list[dict]) -> list["Filename"]:
        return [
            m["filename"]
            for m in filter_images_with_dbdv(matches, self.dbdv_min_id, self.dbdv_max_id)
        ]

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        self._movable_report.move_images()
