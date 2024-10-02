"""Colors for printing."""

from functools import partial

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def colored(color: str, text: str) -> str:
    return f"{color}{text}{ENDC}"


def cprint(color: str, text: str) -> None:
    """Color printing."""
    print(colored(color, text))


def make_cprint(color: str):
    """Make a cprint function with a certain fixed color."""
    return partial(cprint, color)


def make_cprint_with_header(color: str, header_text: str):
    """Make a cprint function with a certain fixed color."""
    return partial(cprint, colored(color, header_text) + " ")
