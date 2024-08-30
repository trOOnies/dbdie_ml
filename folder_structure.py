from os import environ, mkdir
from os.path import isdir, join
from shutil import rmtree

from dotenv import load_dotenv

from dbdie_ml.options import KILLER_FMT, PLAYER_FMT, SURV_FMT
from dbdie_ml.paths import (
    CROP_PENDING_IMG_FD_RP,
    CROPPED_IMG_FD_RP,
    CROPS_MAIN_FD_RP,
    CROPS_VERSIONS_FD_RP,
    IMG_MAIN_FD_RP,
    IMG_VERSIONS_FD_RP,
    IN_CVAT_FD_RP,
    INFERENCE_CROP_PENDING_IMG_FD_RP,
    INFERENCE_CROPPED_IMG_FD_RP,
    INFERENCE_CROPS_MAIN_FD_RP,
    INFERENCE_IMG_MAIN_FD_RP,
    INFERENCE_LABELS_FD_RP,
    INFERENCE_LABELS_MAIN_FD_RP,
    INFERENCE_LABELS_REF_FD_RP,
    LABELS_FD_RP,
    LABELS_MAIN_FD_RP,
    LABELS_REF_FD_RP,
    LABELS_VERSIONS_FD_RP,
    absp,
)

# TODO: Maybe a YAML-oriented implementation?
load_dotenv()


def main():
    print("--------DBDIE FOLDER STRUCTURE CREATION--------")
    print("DBDIE main folder:", environ["DBDIE_MAIN_FD"])
    if isdir(environ["DBDIE_MAIN_FD"]):
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
    else:
        mkdir(environ["DBDIE_MAIN_FD"])

    for f in ["data", "inference"]:
        data_fd = absp(f)
        assert not isdir(data_fd)
        mkdir(data_fd)

    # main subfolders
    mkdir(absp(CROPS_MAIN_FD_RP))
    mkdir(absp(IMG_MAIN_FD_RP))
    mkdir(absp(LABELS_MAIN_FD_RP))
    mkdir(absp(INFERENCE_CROPS_MAIN_FD_RP))
    mkdir(absp(INFERENCE_IMG_MAIN_FD_RP))
    mkdir(absp(INFERENCE_LABELS_MAIN_FD_RP))

    # crops
    all_fmt = list(set(PLAYER_FMT.ALL) | set(KILLER_FMT.ALL) | set(SURV_FMT.ALL))
    for fd in all_fmt:
        mkdir(join(absp(CROPS_MAIN_FD_RP), fd))
        mkdir(join(absp(INFERENCE_CROPS_MAIN_FD_RP), fd))

    # img
    mkdir(absp(CROP_PENDING_IMG_FD_RP))
    mkdir(absp(CROPPED_IMG_FD_RP))
    mkdir(absp(IN_CVAT_FD_RP))
    mkdir(absp(INFERENCE_CROP_PENDING_IMG_FD_RP))
    mkdir(absp(INFERENCE_CROPPED_IMG_FD_RP))

    # labels
    mkdir(absp(LABELS_FD_RP))
    mkdir(absp(LABELS_REF_FD_RP))
    mkdir(absp(INFERENCE_LABELS_FD_RP))
    mkdir(absp(INFERENCE_LABELS_REF_FD_RP))

    # _old_versions
    mkdir(absp(CROPS_VERSIONS_FD_RP))
    mkdir(absp(IMG_VERSIONS_FD_RP))
    mkdir(absp(LABELS_VERSIONS_FD_RP))


if __name__ == "__main__":
    main()
