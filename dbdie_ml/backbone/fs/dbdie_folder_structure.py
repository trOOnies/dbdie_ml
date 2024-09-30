"""DBDIE folder structure helper class."""

from os import environ, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING
import yaml

from dbdie_classes.options.FMT import ALL as ALL_FMTS
from dbdie_classes.options.PLAYER_FMT import ALL as ALL_PLAYER_FMTS

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path, PathToFolder


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

        self.fmts: list["FullModelType"] = ALL_FMTS + ALL_PLAYER_FMTS
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

    def _create_fd(self, sup_fd: "PathToFolder", fd: str | dict, verbose: bool) -> None:
        """Create the folder according to the case:
        - str: Creates the folder and handles the special case of '<FMTs>'.
        - list[str]: Loops this function for each str in the list.
        - dict: Loop inside its contents and apply this function recursively.
        """
        if verbose:
            print("-", sup_fd, "->", fd)

        if isinstance(fd, str):
            assert "." not in fd  # TODO: more restrictive regex
            if fd == "<FMTs>":
                self._create_fmts(sup_fd)
            else:
                mkdir(join(sup_fd, fd))
        elif isinstance(fd, list):
            assert all(isinstance(f, str) for f in fd), "List folder can only be list[str]"
            for f in fd:
                self._create_fd(sup_fd=sup_fd, fd=f, verbose=verbose)
        elif isinstance(fd, dict):
            assert all("." not in k for k in fd)  # TODO: more restrictive regex
            for k, v in fd.items():
                new_sup_fd = join(sup_fd, k)
                mkdir(new_sup_fd)
                self._create_fd(sup_fd=new_sup_fd, fd=v, verbose=verbose)
        else:
            print(fd)
            raise TypeError(f"Folder must be str/list[str]/dict, got {type(fd).__name__}")

    def create_fs(self, verbose: bool) -> None:
        """Create DBDIE folder structure."""
        assert isdir(environ["DBDIE_MAIN_FD"])

        for fd_name, fd in self.fs.items():
            sup_fd = f"{environ['DBDIE_MAIN_FD']}/{fd_name}"
            mkdir(sup_fd)
            self._create_fd(sup_fd, fd, verbose=verbose)

        print("--------DBDIE FOLDER STRUCTURE CREATED--------")
