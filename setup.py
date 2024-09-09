import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict

import tomllib


class QgisPluginBuilder:
    settings: Dict[str, Any]

    def __init__(self):
        current_directory = Path(__file__).parent
        pyproject_file = current_directory / "pyproject.toml"
        self.settings = tomllib.loads(pyproject_file.read_text())

    def prepare(self) -> None:
        self.__compile_resources()
        self.__compile_ui()
        self.compile_ts()

    def update_ts(self):
        ts_settings = (
            self.settings.get("tool", {})
            .get("qgspb", {})
            .get("translation", {})
        )

        command_args = ["pylupdate5"]
        if ts_settings.get("noobsolete", False):
            command_args.append("-noobsolete")

        if ts_settings.get("project-file") is not None:
            command_args.append(
                str(Path(__file__).parent / ts_settings.get("project-file"))
            )
        else:
            source_files = ts_settings.get("source-files", [])
            ts_files = ts_settings.get("ts-files", [])
            if len(source_files) == 0 or len(ts_files) == 0:
                raise RuntimeError("Empty list")

            command_args.extend(
                str(source_path)
                for source_pattern in source_files
                for source_path in Path(__file__).parent.glob(source_pattern)
            )
            command_args.append("-ts")
            command_args.extend(
                str(ts_path)
                for ts_pattern in ts_files
                for ts_path in Path(__file__).parent.glob(ts_pattern)
            )

        subprocess.run(command_args)

        print("TS files have been updated!")

        # TODO (ivanbarsukov): check unfinished in ts files

    def compile_ts(self):
        ts_settings = (
            self.settings.get("tool", {})
            .get("qgspb", {})
            .get("translation", {})
        )
        ts_files = ts_settings.get("ts-files", [])
        command_args = ["lrelease"]
        command_args.extend(
            str(ts_path)
            for ts_pattern in ts_files
            for ts_path in Path(__file__).parent.glob(ts_pattern)
        )

        subprocess.run(command_args)

    def __compile_ui(self) -> None:
        ui_settings = (
            self.settings.get("tool", {}).get("qgspb", {}).get("ui", {})
        )
        if not ui_settings.get("compile", False):
            return

        prefix = ui_settings.get("prefix", "")
        ui_patterns = ui_settings.get("ui-files", [])
        ui_files = [
            ui_path
            for ui_pattern in ui_patterns
            for ui_path in Path(__file__).parent.glob(ui_pattern)
        ]
        for ui_file in ui_files:
            output_directory = ui_file.parent
            output_file = output_directory / f"{prefix}{ui_file.stem}.py"
            subprocess.run(["pyuic5", "-o", str(output_file), str(ui_file)])

    def __compile_resources(self) -> None:
        rc_settings = (
            self.settings.get("tool", {}).get("qgspb", {}).get("resources", {})
        )
        if not rc_settings.get("compile", False):
            return

        suffix = rc_settings.get("suffix", "")
        rc_patterns = rc_settings.get("resource-files", [])
        rc_files = [
            rc_path
            for rc_pattern in rc_patterns
            for rc_path in Path(__file__).parent.glob(rc_pattern)
        ]
        for rc_file in rc_files:
            output_directory = rc_file.parent
            output_file = output_directory / f"{rc_file.stem}{suffix}.py"
            subprocess.run(["pyrcc5", "-o", str(output_file), str(rc_file)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plugin Operations")
    parser.add_argument(
        "action",
        choices=["prepare", "update_ts", "compile_ts"],
        help="Action to perform",
    )

    args = parser.parse_args()
    plugin = QgisPluginBuilder()

    if args.action == "prepare":
        plugin.prepare()
    elif args.action == "update_ts":
        plugin.update_ts()
    elif args.action == "compile_ts":
        plugin.compile_ts()


if __name__ == "__main__":
    main()
