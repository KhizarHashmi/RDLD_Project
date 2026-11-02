"""Make the vendored RoDLA perturbation code and ocrodeg importable.

RoDLA's perturbation modules import each other as top-level modules
(`from blur import ...`) and noise.py does `import ocrodeg`, so both folders
go on sys.path. Only the image-only modules are imported here: spatial.py and
apply_perturbations.py need detectron2 and are out of scope (they move boxes).
"""
import sys

from rdld.config import THIRD_PARTY

_PATHS = [THIRD_PARTY / "ocrodeg", THIRD_PARTY / "rodla_perturbation"]


def enable():
    for p in _PATHS:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
