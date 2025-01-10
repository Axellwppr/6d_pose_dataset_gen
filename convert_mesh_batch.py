# 批量将glb文件转换为isaacsim可读取的usd模型文件
# 该脚本从Isaaclab官方提供的转换脚本修改而来
import argparse
import os
import sys
import glob

# Import Isaac Sim modules
from omni.isaac.lab.app import AppLauncher


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch Utility to convert all GLB files in a directory to USD format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="The path to the input directory containing .glb files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The path to the output directory to store the .usd files.",
    )
    parser.add_argument(
        "--make-instanceable",
        action="store_true",
        default=False,
        help="Make the assets instanceable for efficient cloning.",
    )
    parser.add_argument(
        "--collision-approximation",
        type=str,
        default="convexHull",
        choices=["convexDecomposition", "convexHull", "none"],
        help=(
            "Method used for approximating collision mesh. Choices: "
            '"convexDecomposition", "convexHull", "none". '
            'Set to "none" to not add a collision mesh.'
        ),
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Mass (in kg) to assign to the converted assets. If not provided, no mass is added.",
    )
    # Append AppLauncher CLI args if needed
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_arguments()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app

from omni.isaac.lab.sim.converters import MeshConverter, MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas_cfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

import contextlib


def main():
    # Validate input directory
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Find all .glb files in the input directory
    glb_files = glob.glob(os.path.join(input_dir, "*/textured_simple.obj"))
    if not glb_files:
        print(
            f"No .glb files found in the input directory: {input_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(glb_files)} .glb files to convert.")

    try:
        for glb_path in glb_files:
            try:
                # Get base filename without extension
                base_name = os.path.basename(os.path.dirname(glb_path)).split("_")[0]
                usd_path = os.path.join(output_dir, base_name, f"model.usd")

                # Check if the GLB file exists
                if not check_file_path(glb_path):
                    print(f"Skipping invalid file path: {glb_path}", file=sys.stderr)
                    continue

                # Mass properties
                if args.mass is not None:
                    mass_props = schemas_cfg.MassPropertiesCfg(mass=args.mass)
                    rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
                else:
                    mass_props = None
                    rigid_props = None

                # Collision properties
                collision_props = schemas_cfg.CollisionPropertiesCfg(
                    collision_enabled=args.collision_approximation != "none"
                )

                # Create Mesh converter config
                mesh_converter_cfg = MeshConverterCfg(
                    mass_props=mass_props,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    asset_path=glb_path,
                    force_usd_conversion=True,
                    usd_dir=os.path.join(output_dir, base_name),
                    usd_file_name="model.usd",
                    make_instanceable=args.make_instanceable,
                    collision_approximation=args.collision_approximation,
                    scale=(1.0, 1.0, 1.0),  # Adjust scale if necessary
                )

                # Print conversion info
                print("-" * 80)
                print(f"Converting: {glb_path}")
                print("Mesh importer config:")
                print_dict(mesh_converter_cfg.to_dict(), nesting=0)
                print("-" * 80)

                # Create Mesh converter and perform conversion
                mesh_converter = MeshConverter(mesh_converter_cfg)
                print(f"Generated USD file: {mesh_converter.usd_path}")
                print("-" * 80)

            except Exception as e:
                print(f"Failed to convert {glb_path}: {e}", file=sys.stderr)

        # Determine if there is a GUI to update:
        carb_settings_iface = carb.settings.get_settings()
        local_gui = carb_settings_iface.get("/app/window/enabled")
        livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

        # If GUI is enabled, open the stage and run the app
        if local_gui or livestream_gui:
            stage_utils.open_stage(output_dir)  # You may need to adjust this
            app = omni.kit.app.get_app_interface()
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    app.update()

    finally:
        # Ensure the simulation app is closed properly
        simulation_app.close()


if __name__ == "__main__":
    main()
