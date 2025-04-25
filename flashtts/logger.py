# -*- coding: utf-8 -*-
# Time      :2025/3/18 16:50
# Author    :Hui Huang
import logging
import os
import sys
import threading
from typing import Optional

PROJECT_NAME = "Fast-Spark-TTS"
log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_lock = threading.Lock()
_default_log_level = logging.INFO
_default_handler: Optional[logging.Handler] = None


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(PROJECT_NAME)


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_log_level)

        log_format = f"[{PROJECT_NAME}] %(asctime)s [%(levelname)s] [%(module)s:%(lineno)s] >> %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)
        _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = PROJECT_NAME

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(logging.INFO)


def set_verbosity_warning():
    return set_verbosity(logging.WARNING)


def set_verbosity_debug():
    return set_verbosity(logging.DEBUG)


def set_verbosity_error():
    return set_verbosity(logging.ERROR)


def disable_default_handler() -> None:
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        log_format = f"[{PROJECT_NAME}] %(asctime)s [%(levelname)s] [%(module)s:%(lineno)s] >> %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)
        handler.setFormatter(formatter)


def reset_format() -> None:
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def setup_logging(should_log: bool = True):
    # Setup logging
    logging.basicConfig(
        format=f"[{PROJECT_NAME}] %(asctime)s [%(levelname)s] [%(module)s] >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if should_log:
        set_verbosity_info()

    log_level = logging.INFO if should_log else logging.ERROR
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
