# -*- mode: python ; coding: utf-8 -*-
"""Windows app: double-click Lentic.exe to open the design window."""

from pathlib import Path

root = Path(SPECPATH)

a = Analysis(
    [str(root / "lentic" / "__main__.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "lentic" / "web"), "web")],
    hiddenimports=["manifold3d"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Lentic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
