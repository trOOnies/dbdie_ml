import os
from dbdie_ml.options import KILLER_FMT, SURV_FMT
from dbdie_ml.paths import (
    absp,
    CROPS_MAIN_FD_RP,
    CROPS_VERSIONS_FD_RP,
    IMG_MAIN_FD_RP,
    IMG_VERSIONS_FD_RP,
    CROP_PENDING_IMG_FD_RP,
    CROPPED_IMG_FD_RP,
    IN_CVAT_FD_RP,
    LABELS_MAIN_FD_RP,
    LABELS_VERSIONS_FD_RP,
    LABELS_FD_RP,
    LABELS_REF_FD_RP,
)

print("DBDIE main folder:", os.environ["DBDIE_MAIN_FD"])

data_fd = absp("data")
assert not os.path.isdir(data_fd)
os.mkdir(data_fd)

# main subfolders
os.mkdir(absp(CROPS_MAIN_FD_RP))
os.mkdir(absp(IMG_MAIN_FD_RP))
os.mkdir(absp(LABELS_MAIN_FD_RP))

# crops
all_fmt = list(set(KILLER_FMT.ALL) | set(SURV_FMT.ALL))
for fd in all_fmt:
    os.mkdir(os.path.join(CROPS_MAIN_FD_RP, fd))

# img
os.mkdir(CROP_PENDING_IMG_FD_RP)
os.mkdir(CROPPED_IMG_FD_RP)
os.mkdir(IN_CVAT_FD_RP)

# labels
os.mkdir(LABELS_FD_RP)
os.mkdir(LABELS_REF_FD_RP)

# _old_versions
os.mkdir(CROPS_VERSIONS_FD_RP)
os.mkdir(IMG_VERSIONS_FD_RP)
os.mkdir(LABELS_VERSIONS_FD_RP)
