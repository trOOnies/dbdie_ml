"""Standard paths for DBDIE.

Holds the instatiation of DBDIEFolderStructure (variable 'dbdie_fs').
"""

from os import environ, mkdir
from os.path import isdir, join, relpath, dirname
from shutil import rmtree
from typing import TYPE_CHECKING

import yaml

from dbdie_ml.options import KILLER_FMT, PLAYER_FMT, SURV_FMT

if TYPE_CHECKING:
    from dbdie_ml.classes.base import FullModelType, Path, PathToFolder, RelPath


def absp(rel_path: "RelPath") -> "Path":
    """Convert DBDIE relative path to absolute path"""
    return join(environ["DBDIE_MAIN_FD"], rel_path)


def relp(abs_path: "Path") -> "RelPath":
    """Convert DBDIE absolute path to relative path"""
    return relpath(abs_path, environ["DBDIE_MAIN_FD"])


def recursive_dirname(path: "Path", n: int) -> "PathToFolder":
    """os.path's dirname function but recursive."""
    if n == 1:
        return dirname(path)
    elif n > 1:
        return recursive_dirname(dirname(path), n - 1)
    else:
        raise ValueError(f"n={n} is not a positive integer.")


class DBDIEFolderStructure:
    """DBDIE folder structure helper class.

    Usage:
    >>> dbdie_fs = DBDIEFolderStructure("fs.yaml")
    >>> dbdie_fs.create_main_fd()
    >>> dbdie_fs.create_fs()
    """

    def __init__(self, path: "Path"):
        assert path[-5:] == ".yaml"
        assert ".." not in path[:-5]
        with open(path) as f:
            self.fs = yaml.safe_load(f)

        self.fmts: list["FullModelType"] = list(
            set(PLAYER_FMT.ALL) | set(KILLER_FMT.ALL) | set(SURV_FMT.ALL)
        )
        assert all("." not in fmt for fmt in self.fmts)

    def validate_rp(self, rp: "RelPath") -> "RelPath":
        """Validate if the relative path exists and return it if so."""
        assert isdir(absp(rp)), f"Relative path doesn't exist: {rp}"
        return rp

    def create_main_fd(self) -> None:
        """Create DBDIE folder structure's main folder."""
        print("--------DBDIE FOLDER STRUCTURE CREATION--------")
        print("DBDIE main folder:", environ["DBDIE_MAIN_FD"])

        if not isdir(environ["DBDIE_MAIN_FD"]):
            mkdir(environ["DBDIE_MAIN_FD"])
        else:
            while True:
                ans = input(
                    "Main folder already exists. Do you want to recreate it? (y/n): "
                )
                if ans.lower() in {"y", "yes"}:
                    rmtree(environ["DBDIE_MAIN_FD"])
                    mkdir(environ["DBDIE_MAIN_FD"])
                    break
                elif ans.lower() in {"n", "no"}:
                    print("Stopping folder structure creation...")
                    return
                else:
                    print("Invalid option.")

    def _create_fmts(self, fd_path: "PathToFolder") -> None:
        """Create subfolders from FullModelTypes."""
        assert all("." not in fmt for fmt in self.fmts)  # TODO: more restrictive regex
        for fmt in self.fmts:
            mkdir(join(fd_path, fmt))

    def _create_fd(self, sup_fd: "PathToFolder", fd: str | dict) -> None:
        """Create the folder if it's a str, or keep looping if it's a dict."""
        if isinstance(fd, str):
            assert "." not in fd  # TODO: more restrictive regex
            if fd == "<FMTs>":
                self._create_fmts(sup_fd)
            else:
                mkdir(join(sup_fd, fd))
        elif isinstance(fd, dict):
            assert all("." not in k for k in fd)  # TODO: more restrictive regex
            for k, v in fd.items():
                new_sup_fd = join(sup_fd, k)
                mkdir(new_sup_fd)
                self._create_fd(sup_fd=new_sup_fd, fd=v)
        else:
            raise TypeError("Folder must be either str or a dict")

    def create_fs(self) -> None:
        """Create DBDIE folder structure."""
        assert isdir(environ["DBDIE_MAIN_FD"])

        self._create_fd(environ["DBDIE_MAIN_FD"], self.fs["data"])
        self._create_fd(environ["DBDIE_MAIN_FD"], self.fs["inference"])


dbdie_fs = DBDIEFolderStructure(join(dirname(__file__), "configs/fs.yaml"))
vrp = dbdie_fs.validate_rp

# * Paths

OLD_VS = "_old_versions"

# * Training

CROPS_MAIN_FD_RP = vrp("data/crops")
CROPS_VERSIONS_FD_RP = vrp(f"data/crops/{OLD_VS}")

IMG_MAIN_FD_RP = vrp("data/img")
CROP_PENDING_IMG_FD_RP = vrp("data/img/pending")
CROPPED_IMG_FD_RP = vrp("data/img/cropped")
IMG_VERSIONS_FD_RP = vrp(f"data/img/{OLD_VS}")


LABELS_MAIN_FD_RP = vrp("data/labels")
LABELS_FD_RP = vrp("data/labels/labels")
LABELS_REF_FD_RP = vrp("data/labels/label_ref")
LABELS_VERSIONS_FD_RP = vrp(f"data/labels/{OLD_VS}")

# * Inference

INFERENCE_CROPS_MAIN_FD_RP = vrp("inference/crops")

INFERENCE_IMG_MAIN_FD_RP = vrp("inference/img")
INFERENCE_CROP_PENDING_IMG_FD_RP = vrp("inference/img/pending")
INFERENCE_CROPPED_IMG_FD_RP = vrp("inference/img/cropped")

INFERENCE_LABELS_MAIN_FD_RP = vrp("inference/labels")
INFERENCE_LABELS_FD_RP = vrp("inference/labels/labels")
INFERENCE_LABELS_REF_FD_RP = vrp("inference/labels/label_ref")
