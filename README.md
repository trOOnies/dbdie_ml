# DBD Info Extraction - ML package

<center>💀 <i>Information extraction for Dead By Daylight (game) end scorecard.</i> 💀</center>

## dbdie_ml

This repo contains the code necessary for the actual ML models and their wrappers for ease-of-use. It's designed to only have the classes I'll import in more complex pipelines, so it can be installed via `pip` without all the overhead I may add in repos like [dbdie_info_extraction](https://github.com/trOOnies/dbd_info_extraction) (WIP).

It is very much a WIP repo. I write it whenever I'm feeling inspired, thus its development is slow and erratic, ngl, but you are free to expand on it according to the licence's permissions and restrictions.

## Classes

- `Cropper`: Crops images or image snippets according to its settings. Used to avoid hardcoding settings in the code.
- `CropperSwarm`: Chain of `Croppers` that can be run in sequence.
- `InfoExtractor`: Extracts information of an image using multiple `IEModels`.
- `IEModel`: ML model for the extraction of a particular information type, such as perks, addons, etc.
- `MatchOut`: Extracted match information.

## Environment variables

You must set an env var `DBDIE_MAIN_FD` that points to a folder with the following folder structure:

```
<DBDIE_MAIN_FD>
└── data
    ├── crops
    │   ├── _old_versions
    │   │   ├── ...
    │   │   └── 0
    │   │       ├── addons__killer
    │   │       ├── ...
    │   │       └── status
    │   ├── addons__killer
    │   ├── addons__surv
    │   ├── character__killer
    │   ├── character__surv
    │   ├── item__killer
    │   ├── item__surv
    │   ├── offering__killer
    │   ├── offering__surv
    │   ├── perks__killer
    │   ├── perks__surv
    │   ├── player__killer
    │   ├── player__surv
    │   ├── points
    │   ├── prestige
    │   └── status
    ├── img
    │   ├── _old_versions
    │   │   ├── ...
    │   │   └── 0
    │   │       ├── cropped
    │   │       ├── in_cvat
    │   │       ├── pending
    │   │       └── pending_not_std
    │   ├── cropped
    │   ├── in_cvat
    │   ├── pending
    │   └── pending_not_std
    ├── labels
    │   ├── _old_versions
    │   │   ├── ...
    │   │   └── 0
    │   │       ├── labels
    │   │       │   ├── perks__killer
    │   │       │   └── perks__surv
    │   │       ├── (ex.) project_dbdie_perks__killer-2023_11_20_19_58_04-cvat for images 1.1
    │   │       └── (ex.) project_dbdie_perks__surv-2023_11_21_03_14_26-cvat for images 1.1
    │   ├── label_ref
    │   ├── labels
    │   │   ├── character__killer
    │   │   ├── character__surv
    │   │   ├── perks__killer
    │   │   └── perks__surv
    │   ⋮
    │   └── (ex.) project_dbdie_character__killer-2024_07_10_22_25_03-cvat for images 1.1
    └── out
```

Code's still in development, but there's a plan to make this folder structure automatically.

## Open configuration

If you just want to know how do I crop the endcard screenshot and the Pytorch architectures I use, check out the following files:

- dbdie_ml/crop_settings.py
- dbdie_ml/models/custom.py

## License

I chose to link all the code I develop to a **GPL-3.0** license. You can see its details in the `LICENSE` file, but I find it easier to read its summary [here](https://choosealicense.com/licenses/gpl-3.0/).

## See also

- [dbdie_api](https://github.com/trOOnies/dbdie_api)
- [dbdie_info_extraction](https://github.com/trOOnies/dbd_info_extraction)
- [dbdie_info_training](https://github.com/trOOnies/dbd_info_training)
