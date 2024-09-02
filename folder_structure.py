"""Create DBDIE folder structure"""

from dotenv import load_dotenv

from dbdie_ml.paths import dbdie_fs

load_dotenv()


def main():
    dbdie_fs.create_main_fd()
    dbdie_fs.create_fs()


if __name__ == "__main__":
    main()
