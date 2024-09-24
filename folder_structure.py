"""Script: create DBDIE folder structure."""

from os.path import dirname, join
from dotenv import load_dotenv

load_dotenv()

from dbdie_ml.fs.dbdie_folder_structure import DBDIEFolderStructure  # noqa: E402


def main():
    path = join(dirname(__file__), "dbdie_ml/configs/folder_structure.yaml")
    dbdie_fs = DBDIEFolderStructure(path)
    dbdie_fs.create_main_fd()
    dbdie_fs.create_fs(verbose=True)


if __name__ == "__main__":
    main()
