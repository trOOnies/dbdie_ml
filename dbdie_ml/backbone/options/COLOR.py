"""Colors for printing."""

from functools import partial

ENDC      = "\033[0m"
BOLD      = "\033[1m"
UNDERLINE = "\033[4m"

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

BRIGHT_BLACK   = "\033[90m"
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

# Background colors
BG_BLACK   = "\033[40m"
BG_RED     = "\033[41m"
BG_GREEN   = "\033[42m"
BG_YELLOW  = "\033[43m"
BG_BLUE    = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN    = "\033[46m"
BG_WHITE   = "\033[47m"

PER_CLASS = {
    "MovableReport": BRIGHT_RED,
    "CropperSwarm": BRIGHT_MAGENTA,
    "IEModel": BRIGHT_BLUE,
    "InfoExtractor": BRIGHT_CYAN,
}


def colored(color: str, text: str) -> str:
    """Return colored text."""
    return f"{color}{text}{ENDC}"


def cprint(color: str, text: str) -> None:
    """Print text using `color`."""
    print(colored(color, text))


def make_cprint(color: str):
    """Make a cprint function with a certain fixed `color`."""
    return partial(cprint, color)


def make_cprint_with_header(color: str, header_text: str):
    """Make a header-based cprint function with a certain fixed `color`."""
    return partial(cprint, colored(color, header_text) + " ")


def get_class_cprint(name: str):
    """Get the corresponding DBDIE class' cprint function."""
    return partial(cprint, colored(PER_CLASS[name], f"[{name}]") + " ")
