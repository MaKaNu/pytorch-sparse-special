"""
Error module: provides the error classes for pytorch_sparse_special.
    Copyright (C) 2025  MaKaNU

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any


class SizeValueError(ValueError):
    def __init__(self, obj: Any) -> None:
        super().__init__(f"{type(obj)} is defined as 3D Matrix. Fix size or indices attribute!")
