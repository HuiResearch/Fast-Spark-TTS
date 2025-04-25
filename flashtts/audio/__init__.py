# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:25
# Author    :Hui Huang
from ..import_utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "tokenizer": [
        "SparkTokenizer",
        "MegaTokenizer"
    ],
    "detokenizer": [
        'SparkDeTokenizer',
        "SnacDeTokenizer"
    ]
}
if TYPE_CHECKING:
    from .tokenizer import (
        SparkTokenizer,
        MegaTokenizer
    )
    from .detokenizer import (
        SparkDeTokenizer,
        SnacDeTokenizer
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
