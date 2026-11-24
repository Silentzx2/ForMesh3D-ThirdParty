"""sitecustomize for SkinTokens.

Auto-imported by Python at startup when backend/thirdparty/SkinTokens/ is on
sys.path (it is, because bpy_server.py is launched from that directory).

Injects stub modules for the missing Michelangelo models package so that
bpy_server.py (a subprocess of demo.start_bpy_server()) can import
src.model.michelangelo.models.tsal.sal_perceiver without the full upstream
source vendored. This only affects the SkinTokens worker subprocess.
"""
from __future__ import annotations

import sys
import types

_MOD_NAME = "src.model.michelangelo.models.tsal.sal_perceiver"

if _MOD_NAME not in sys.modules:
    # Build the missing package chain so `import src.model.michelangelo.models...`
    # resolves when skintokens_root is on sys.path.
    for _pkg_name in (
        "src",
        "src.model",
        "src.model.michelangelo",
        "src.model.michelangelo.models",
        "src.model.michelangelo.models.tsal",
    ):
        if _pkg_name not in sys.modules:
            _pkg = types.ModuleType(_pkg_name)
            _pkg.__path__ = []
            _pkg.__package__ = _pkg_name
            sys.modules[_pkg_name] = _pkg

    _mod = types.ModuleType(_MOD_NAME)

    class AlignedShapeLatentPerceiver:
        """Stub for the real Michelangelo encoder. Not vendored upstream."""
        _skintokens_stub = True

        def __init__(self, **kwargs):
            pass

        def load_state_dict(self, state_dict, strict=True):
            return self

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "SkinTokens: real Michelangelo models source is not vendored. "
                "Vendor backend/thirdparty/SkinTokens/src/model/michelangelo/models/ and retry."
            )

    class ShapeAsLatentPerceiverEncoder:
        """Stub for the real Michelangelo encoder. Not vendored upstream."""
        _skintokens_stub = True

        def __init__(self, **kwargs):
            pass

        def load_state_dict(self, state_dict, strict=True):
            return self

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "SkinTokens: real Michelangelo models source is not vendored. "
                "Vendor backend/thirdparty/SkinTokens/src/model/michelangelo/models/ and retry."
            )

    _mod.AlignedShapeLatentPerceiver = AlignedShapeLatentPerceiver
    _mod.ShapeAsLatentPerceiverEncoder = ShapeAsLatentPerceiverEncoder
    _mod.__package__ = "src.model.michelangelo.models.tsal"
    sys.modules[_MOD_NAME] = _mod
