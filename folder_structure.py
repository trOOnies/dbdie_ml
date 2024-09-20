"""Script: create DBDIE folder structure."""

from os.path import dirname, join
from dotenv import load_dotenv

from dbdie_ml.classes.paths import DBDIEFolderStructure

load_dotenv()


def main():
    dbdie_fs = DBDIEFolderStructure(
        join(dirname(__file__), "dbdie_ml/configs/folder_structure.yaml")
    )
    dbdie_fs.create_main_fd()
    dbdie_fs.create_fs()


if __name__ == "__main__":
    main()
