# -*- coding: utf-8 -*-
# Time      :2025/3/14 20:24
# Author    :Hui Huang

from .import_utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "engine": [
        "BaseEngine",
        "AsyncSparkEngine",
        "AsyncOrpheusEngine",
        "AsyncMega3Engine",
        "SparkAcousticTokens",
        "AutoEngine"
    ],
    "logger": [
        "get_logger",
        "setup_logging"
    ]
}

if TYPE_CHECKING:
    from .logger import get_logger, setup_logging
    from .engine import (
        BaseEngine,
        AsyncSparkEngine,
        AsyncOrpheusEngine,
        SparkAcousticTokens,
        AsyncMega3Engine,
        AutoEngine
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
