# DBD Info Extraction - ML package

💀 Information extraction for Dead By Daylight (game) end scorecard. 💀

This repo contains the code necessary for the actual ML models and their wrappers for ease-of-use. It's designed to only have the classes I'll import in more complex pipelines, so it can be installed via `pip` without all the overhead I may add in repos like [dbdie_info_extraction](https://github.com/trOOnies/dbd_info_extraction) (WIP).

It is very much a WIP repo. I write it whenever I'm feeling inspired, thus its development is slow and erratic, ngl, but you are free to expand on it according to the licence's permissions and restrictions.

## Classes

- `InfoExtractor`: Extracts information of an image using multiple `IEModels`.
- `IEModel`: ML model for the extraction of a particular information type, such as perks, addons, etc.
- `MatchOut`: Extracted match information.

## License

I chose to link all the code I develop to a **GPL-3.0** license. You can see its details in the `LICENSE` file, but I find it easier to read its summary [here](https://choosealicense.com/licenses/gpl-3.0/).
