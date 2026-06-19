#!/usr/bin/env python3
"""Add rigid-body physics to existing USDC files.

Run:  ./isaaclab.sh -p /workspace/hirol/dependencies/isaaclab/local_factory_assets/add_physics.py --headless
"""

from pathlib import Path

from isaaclab.app import AppLauncher

# Must go through AppLauncher to get pxr access
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf

ASSETS_DIR = Path("/workspace/hirol/dependencies/isaaclab/local_factory_assets")
FILES = ["USBA_plug.usdc", "USBA_socket.usdc"]


def add_color(mesh_prim, color=(0.5, 0.5, 0.5)):
    """Set displayColor on mesh — no USD material/shader needed."""
    from pxr import Vt
    geom = UsdGeom.Mesh(mesh_prim)
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def main():
    for fname in FILES:
        path = str(ASSETS_DIR / fname)
        print(f"Processing {path}...")
        stage = Usd.Stage.Open(path)

        count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                p = prim.GetPath()
                print(f"  Mesh: {p}")
                # ArticulationRootAPI: required by Isaac Lab Articulation class
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.CollisionAPI.Apply(prim)
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                if mass_api:
                    UsdPhysics.MassAPI(prim).CreateMassAttr().Set(0.01)
                add_color(prim)
                count += 1

        # Also handle Xform (single top-level prim in converted STL)
        if count == 0:
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xform):
                    has_mesh = any(c.IsA(UsdGeom.Mesh) for c in prim.GetAllChildren())
                    if has_mesh:
                        p = prim.GetPath()
                        print(f"  Xform (has mesh): {p}")
                        UsdPhysics.ArticulationRootAPI.Apply(prim)
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                        UsdPhysics.CollisionAPI.Apply(prim)
                        mass_api = UsdPhysics.MassAPI.Apply(prim)
                        if mass_api:
                            UsdPhysics.MassAPI(prim).CreateMassAttr().Set(0.01)
                        add_color(prim)
                        count += 1

        stage.Save()
        print(f"  Added physics to {count} prims → saved")
    print("Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
