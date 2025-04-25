# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:04
# Author    :Hui Huang
from ..import_utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "base_engine": [
        "BaseEngine"
    ],
    "spark_engine": ["AsyncSparkEngine", "SparkAcousticTokens"],
    "orpheus_engine": ["AsyncOrpheusEngine"],
    "mega_engine": ["AsyncMega3Engine"],
    "auto_engine": ["AutoEngine"]
}

if TYPE_CHECKING:
    from .base_engine import BaseEngine
    from .spark_engine import AsyncSparkEngine, SparkAcousticTokens
    from .orpheus_engine import AsyncOrpheusEngine
    from .mega_engine import AsyncMega3Engine
    from .auto_engine import AutoEngine
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
