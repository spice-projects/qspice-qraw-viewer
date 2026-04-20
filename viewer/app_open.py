from collections.abc import Callable
from pathlib import Path

from .qraw_file import QRawFile


def open_qraw_as_window(input_path: str | Path, window_factory: Callable[[QRawFile], object]) -> object | None:
    # load and parse the QRAW file
    qraw_file = QRawFile.load(input_path)
    # return none when parsing fails
    if qraw_file is None:
        return None
    # create the target window from the loaded qraw object
    return window_factory(qraw_file)
