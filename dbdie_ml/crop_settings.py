from dbdie_ml.classes import CropSettings, DBDVersionRange
from dbdie_ml.constants import DEFAULT_DBDVR, DEFAULT_SCREENSHOT_SIZE
from dbdie_ml.options import CROP_TYPES
from dbdie_ml.paths import CROP_PENDING_IMG_FD_RP, CROPS_MAIN_FD_RP

IMG_SURV_CS = CropSettings(
    name=CROP_TYPES.SURV,
    src_fd_rp=CROP_PENDING_IMG_FD_RP,
    dst_fd_rp=CROPS_MAIN_FD_RP,
    version_range=DBDVersionRange(*DEFAULT_DBDVR),
    img_size=DEFAULT_SCREENSHOT_SIZE,
    crops={
        "player__surv": [
            (67, 257 + j * 117, 897, 257 + (j + 1) * 117) for j in range(3)
        ]
        + [(67, 257 + 3 * 117 + 1, 897, 257 + (3 + 1) * 117 + 1)]
    },
)
IMG_KILLER_CS = CropSettings(
    name=CROP_TYPES.KILLER,
    src_fd_rp=CROP_PENDING_IMG_FD_RP,
    dst_fd_rp=CROPS_MAIN_FD_RP,
    version_range=DBDVersionRange(*DEFAULT_DBDVR),
    img_size=DEFAULT_SCREENSHOT_SIZE,
    crops={"player__killer": [(66, 716, 896, 716 + 117)]},
    offset=4,
)

PLAYER_SURV_CS = CropSettings(
    name=CROP_TYPES.SURV_PLAYER,
    src_fd_rp=f"{CROPS_MAIN_FD_RP}/player__surv",
    dst_fd_rp=CROPS_MAIN_FD_RP,
    version_range=DBDVersionRange(*DEFAULT_DBDVR),
    img_size=IMG_SURV_CS.crop_sizes["player__surv"],
    crops={
        "addons__surv": [(483 + j * 41, 58, 483 + (j + 1) * 41, 99) for j in range(2)],
        "character__surv": [(124, 5, 600, 38)],
        "item__surv": [(422, 59, 465, 101)],
        "offering__surv": [(356, 54, 405, 107)],
        "perks__surv": [(123 + j * 55, 50, 123 + (j + 1) * 55, 106) for j in range(4)],
        "points": [(580, 54, 750, 104)],
        "prestige": [(0, 0, 117, 116)],
        "status": [(124, 5, 155, 46)],
    },
)
PLAYER_KILLER_CS = CropSettings(
    name=CROP_TYPES.KILLER_PLAYER,
    src_fd_rp=f"{CROPS_MAIN_FD_RP}/player__killer",
    dst_fd_rp=CROPS_MAIN_FD_RP,
    version_range=DBDVersionRange(*DEFAULT_DBDVR),
    img_size=IMG_KILLER_CS.crop_sizes["player__killer"],
    crops={
        "addons__killer": [
            (483 + j * 41, 67, 483 + (j + 1) * 41, 108) for j in range(2)
        ],
        "character__killer": [(124, 15, 600, 48)],
        "item__killer": [(422, 68, 465, 110)],
        "offering__killer": [(356, 63, 405, 116)],
        "perks__killer": [
            (123 + j * 55, 60, 123 + (j + 1) * 55, 116) for j in range(4)
        ],
        "points": [(582, 64, 752, 114)],
        "prestige": [(0, 9, 117, 125)],
        "status": [(124, 10, 155, 51)],
    },
    offset=4,
)

ALL_CS = [IMG_SURV_CS, IMG_KILLER_CS, PLAYER_SURV_CS, PLAYER_KILLER_CS]
