# DBD Info Extraction - ML package

<center>💀 <i>Information extraction for Dead By Daylight (game) end scorecard.</i> 💀</center>

## Description

This repo contains the code necessary for the actual ML models and their wrappers for ease-of-use. It's designed to only have the classes I'll import in more complex pipelines, so it can be installed via `pip` without all the overhead I may add in repos like [dbdie_info_extraction](https://github.com/trOOnies/dbd_info_extraction) (WIP).

It is very much a WIP repo. I write it whenever I'm feeling inspired, thus its development is slow and erratic, ngl, but you are free to expand on it according to the licence's permissions and restrictions.

## Installation

It is recommended that you have the dbdie_ml and dbdie_api repos at the same folder level, in some path of your choice. Also, the code's now being developed in Linux, so it's best to use it in Linux or in Windows' WSL.

### Option 1

If you want the functionalities the API provides (including the use of the PostgreSQL database), you should navigate to the dbdie_api repo either by CLI or using a code editor like VS Code, and you should install the dbdie_ml dependency.

```bash
pip install ../dbdie_ml
```

For your convenience, there's also a Makefile available that should make this process more straightforward.

```bash
make venv  # creates the .venv folder in the repo's directory
source .venv/bin/activate
make install  # installs external dependencies and the dbdie_ml package
make api  # runs the API on localhost
```

### Option 2

You can explore the code for yourself without the API just by installing the package on a venv of your choice.

```bash
pip install <path_to_dbdie_ml_fd>
```

## Classes

- `Cropper`: Crops images according to its settings. Used to avoid hardcoding settings in the code.
- `CropperSwarm`: Chain of `Croppers` that can be run in sequence.
- `InfoExtractor`: Extracts information of an image using multiple `IEModels`.
- `IEModel`: ML model for the extraction of a particular information type, such as perks, addons, etc.
- `FullMatchOut`: Extracted match information.

## Environment variables

You must set an env var `DBDIE_MAIN_FD` that points to a folder which will contain the DBDIE folder structure.
You can create this structure using the `make folders` command.

Data folder
```bash
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
    │   │       └── pending
    │   ├── cropped
    │   └── pending
    └── labels
        ├── _old_versions
        │   ├── ...
        │   └── 0
        │       ├── labels
        │       │   ├── perks__killer
        │       │   └── perks__surv
        │       ├── (ex.) project_dbdie_perks__killer-2023_11_20_19_58_04-cvat for images 1.1
        │       └── (ex.) project_dbdie_perks__surv-2023_11_21_03_14_26-cvat for images 1.1
        ├── label_ref
        ├── labels
        │   ├── character__killer
        │   ├── character__surv
        │   ├── perks__killer
        │   └── perks__surv
        ⋮
        └── (ex.) project_dbdie_character__killer-2024_07_10_22_25_03-cvat for images 1.1
```

Inference folder (WIP)
```bash
<DBDIE_MAIN_FD>
└── inference
    ├── crops
    │   ├── addons__killer
    │   └── ...
    ├── img
    │   ├── cropped
    │   └── pending
    └── labels
        ├── label_ref
        └── labels
            ├── character__killer
            └── ...
```

## Usage examples

(WIP)

## Open configuration (TLDR)

If you just want to know how do I crop the endcard screenshot and the Pytorch architectures I use, check out the following files:

- dbdie_ml/crop_settings.py
- dbdie_ml/models/custom.py

## Contributing

If you have any images I can use to further train my models with data different than my own, feel free to reach out and help contribute to this project.
Images shared will be used solely for the experimentation and training of this project, won't be uploaded to this or any other place.

You can take the credit in this README if you want to. Also, the user who shared them can at any time revoke their use, and thus the images will be promptly deleted from the ML database.

No labels are needed but they are welcome nonetheless.

## License

I chose to link all the code I develop to a **GPL-3.0** license. You can see its details in the `LICENSE` file, but I find it easier to read its summary [here](https://choosealicense.com/licenses/gpl-3.0/).

## See also

- [dbdie_api](https://github.com/trOOnies/dbdie_api)
- [dbdie_ui](https://github.com/trOOnies/dbdie_ui)
- [dbdie_info_extraction](https://github.com/trOOnies/dbd_info_extraction)
- [dbdie_info_training](https://github.com/trOOnies/dbd_info_training)
