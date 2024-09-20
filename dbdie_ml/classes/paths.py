"""DBDIE folder structure helper class."""

from os import environ, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING

import yaml

from dbdie_ml.options import KILLER_FMT, PLAYER_FMT, SURV_FMT

if TYPE_CHECKING:
    from dbdie_ml.classes.base import FullModelType, Path, PathToFolder


class DBDIEFolderStructure:
    """DBDIE folder structure helper class.

    Usage:
    >>> dbdie_fs = DBDIEFolderStructure("folder_structure.yaml")
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
