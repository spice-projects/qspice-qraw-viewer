import argparse
import logging
import sys
import warnings

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from viewer.main_window import MainWindow
from viewer.qraw_file import QRawFile

logger = logging.getLogger(__name__)

_qml_logger = logging.getLogger("viewer.qml")
_warnings_logger = logging.getLogger("viewer.warnings")

_QT_LEVEL_MAP = {
    QtMsgType.QtDebugMsg:    logging.DEBUG,
    QtMsgType.QtInfoMsg:     logging.INFO,
    QtMsgType.QtWarningMsg:  logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg:    logging.CRITICAL,
}


def _qt_message_handler(msg_type: QtMsgType, _, message: str) -> None:
    # translate level
    level = _QT_LEVEL_MAP.get(msg_type, logging.WARNING)
    # write message to the QML logger
    _qml_logger.log(level, "%s", message)


def _warning_handler(message, category, filename, lineno, file=None, line=None) -> None:
    # write message to the warnings logger
    _warnings_logger.warning("%s:%d: %s: %s", filename, lineno, category.__name__, message)


def main():
    # use Windows-style flag prefix on Windows, UNIX-style on all other platforms
    is_windows = sys.platform == "win32"
    prefix = "/" if is_windows else "--"
    short_prefix = "/" if is_windows else "-"
    # configure argument parser with platform-appropriate prefix character
    parser = argparse.ArgumentParser(description="QSPICE QRAW Viewer", prefix_chars="/" if is_windows else "-")
    # input file is required (always)
    parser.add_argument("input", nargs="?", help="Input QRAW file")
    # headless mode, used to export data
    parser.add_argument(f"{short_prefix}H", f"{prefix}headless", action="store_true", help="Parse and log the file without opening the viewer UI")
    # log level, defaults to WARNING
    parser.add_argument(f"{prefix}log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    # parse command line arguments
    args = parser.parse_args()
    # configure logging with the requested level
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # route all Qt/QML messages through Python logging
    qInstallMessageHandler(_qt_message_handler)
    # route all Python/numpy warnings through Python logging
    warnings.showwarning = _warning_handler
    # validate required input argument
    if not args.input:
        # log information
        logger.error("no input file provided")
        # exit
        sys.exit(1)
    # load and parse the QRAW file
    qraw_file = QRawFile.load(args.input)
    if qraw_file is None:
        # exit
        sys.exit(1)
    # check we need to launch the viewer UI
    if not args.headless:
        # create Qt application
        app = QApplication(sys.argv)
        # main window with the loaded QRAW file
        window = MainWindow(qraw_file)
        # show the main window
        window.show()
        # enter the Qt application main loop
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
