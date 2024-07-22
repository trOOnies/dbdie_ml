def pls(item: str, length: int) -> str:
    """Plural letter 's' friendly count"""
    return f"{length} {item}{'s' if length > 1 else ''}"
