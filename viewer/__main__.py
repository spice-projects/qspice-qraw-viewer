import argparse
import logging
import sys
import warnings

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QFileDialog

from viewer.app_open import open_qraw_as_window
from viewer.main_window import load_app_icon, MainWindow
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
    # configure argument parser
    parser = argparse.ArgumentParser(description="QSPICE QRAW Viewer")
    # input file is optional; when omitted in GUI mode an open-file dialog is shown
    parser.add_argument("input", nargs="?", help="Input QRAW file (required in headless mode; prompts via file dialog if omitted in GUI mode)")
    # headless mode parses and logs the file without opening the viewer UI; requires input
    parser.add_argument("-H", "--headless", action="store_true", help="Parse and log the file without opening the viewer UI (input file required)")
    # log level, defaults to WARNING
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    # parse command line arguments
    args = parser.parse_args()
    # configure logging with the requested level
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # route all Qt/QML messages through Python logging
    qInstallMessageHandler(_qt_message_handler)
    # route all Python/numpy warnings through Python logging
    warnings.showwarning = _warning_handler
    # headless mode requires a file path on the command line
    if args.headless:
        # input is required
        if not args.input:
            # log error
            logger.error("headless mode requires an input file")
            # exit
            sys.exit(1)
        # load and parse the QRAW file
        qraw_file = QRawFile.load(args.input)
        if qraw_file is None:
            # exit
            sys.exit(1)
        # nothing more to do in headless mode
        sys.exit(0)
    # create Qt application before any dialog or window
    app = QApplication(sys.argv)
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    # prompt for a file when none was provided on the command line
    input_path = args.input
    if not input_path:
        # open OS file picker; returns (path, filter)
        input_path, _ = QFileDialog.getOpenFileName(None, "Open QRAW File", "", "QRAW Files (*.qraw);;All Files (*)")
        # exit when the user cancels the dialog
        if not input_path:
            sys.exit(0)
    # load file and create the main window through shared open path
    window = open_qraw_as_window(input_path, lambda qraw_file: MainWindow(qraw_file))
    if window is None:
        # exit
        sys.exit(1)
    # show the main window
    window.show()
    # enter the Qt application main loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
