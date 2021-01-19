"""型を定義"""


from typing import Generic, NamedTuple, TypeVar

import numpy as np

__all__ = [
    # Array
    "ActionArray",
    "ObservationArray",
    # 範囲
    "Range",
]


# ======================================
# Array
# ======================================

# shape=(ac_size, ), dtype=float
ActionArray = np.ndarray

# shape=(ob_size, ), dtype=float
ObservationArray = np.ndarray


# ======================================
# 画像
# ======================================

# shape=(height, width, 3), dtype=numpy.uint8
RGBArray = np.ndarray


# ======================================
# 範囲
# ======================================

_T = TypeVar("_T")

# NOTE
# Rangeを利用するファイルでは
# `from __future__ import annotations`
# を記述すること。
# https://stackoverflow.com/questions/50530959/generic=namedtuple=in=python=3=6
class Range(NamedTuple, Generic[_T]):
    low: _T
    high: _T
