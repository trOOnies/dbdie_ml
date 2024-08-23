from dbdie_ml.classes import CropSettings

IMG_SURV_CS = CropSettings.from_config("img_surv_cs")
IMG_KILLER_CS = CropSettings.from_config("img_killer_cs")
PLAYER_SURV_CS = CropSettings.from_config(
    "player_surv_cs",
    depends_on=IMG_SURV_CS,
)
PLAYER_KILLER_CS = CropSettings.from_config(
    "player_killer_cs",
    depends_on=IMG_KILLER_CS,
)

ALL_CS = [IMG_SURV_CS, IMG_KILLER_CS, PLAYER_SURV_CS, PLAYER_KILLER_CS]