#!/usr/bin/env python3
"""Convert USB STL files to USD format for Isaac Lab Factory tasks.

Run inside Docker:
  python local_factory_assets/convert_usb.py
"""

import os
from pathlib import Path

import omni.kit.asset_converter
import omni.usd

ASSETS_DIR = Path("/workspace/hirol/dependencies/isaaclab/local_factory_assets")
INPUTS = ["USBA_plug.stl", "USBA_socket.stl"]
OUTPUTS = ["USBA_plug.usd", "USBA_socket.usd"]


def convert_stl_to_usd(stl_path: str, usd_path: str):
    """Convert STL to USD using Isaac Sim asset converter."""
    print(f"Converting {stl_path} → {usd_path}")

    # Setup converter
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = False
    converter_context.ignore_animations = True
    converter_context.ignore_cameras = True
    converter_context.ignore_lights = True

    # Convert
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        stl_path, usd_path, converter_context)

    success = task.wait()
    if not success:
        print(f"  FAILED: {task.get_status()}")
        return False

    print(f"  OK → {usd_path}")
    return True


def main():
    for inp, out in zip(INPUTS, OUTPUTS):
        stl = str(ASSETS_DIR / inp)
        usd = str(ASSETS_DIR / out)
        if not os.path.exists(stl):
            print(f"SKIP {stl} (not found)")
            continue
        convert_stl_to_usd(stl, usd)
    print("Done.")


if __name__ == "__main__":
    main()
