"""The Java source trees this harness is pointed at, and what running one proves.

A Task is the harness's input port. `instances()` lists one directory per source tree, `setup` copies that tree into the run directory, `prompt` says what the item's agent is asked for, and `verify` runs the build script the pass wrote and compares what the program printed with the output recorded beside the sources.

An item directory has three parts, and they are what a person needs to add a tree of their own:

    <item>/src/**.java    the sources
    <item>/item.json      {"main_class": "...", "main_args": "..."}
    <item>/expected.txt   exactly what running the entry point prints

These three are the fixture the package ships. To run the harness on your own trees, point the run at them (`refactory ceo java_build_pack/build --source <dir>`): the items a harness reads are run configuration, so a Task does not need a second mechanism for it and the graph never carries them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from refactory_dsh.task import Task, TaskInstance, VerifyResult

#: The item directory a checkout ships, beside this module.
ITEMS = Path(__file__).resolve().parent / "items"

#: How long one build script may run before the pass is judged a failure.
BUILD_TIMEOUT_S = 120


def _javac_works(bin_dir: Path) -> bool:
    """Whether `bin_dir/javac` is a working compiler rather than a stub.

    macOS ships `/usr/bin/javac` as a launcher that exits non-zero when no JDK is installed, so a JDK is only present once `javac -version` says so.
    """
    executable = bin_dir / "javac"
    if not executable.exists():
        return False
    try:
        completed = subprocess.run(
            [str(executable), "-version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def java_bin() -> Path | None:
    """The directory holding a working `javac`, or None when this machine has no JDK.

    The PATH entry comes first, so a machine whose JDK is already exported uses it. The remaining locations cover one where the JDK is installed but not on PATH, which is how Homebrew installs OpenJDK; `refactory_dsh.paths.node_bin` resolves Node the same way. `JavaBuildTask.verify` puts this directory on the PATH the build script runs with, so a script that invokes `javac` and `java` finds the toolchain.
    """
    candidates: list[Path] = []
    found = shutil.which("javac")
    if found:
        candidates.append(Path(found).resolve().parent)
    home = os.environ.get("JAVA_HOME", "").strip()
    if home:
        candidates.append(Path(home) / "bin")
    candidates += [
        Path("/opt/homebrew/opt/openjdk/bin"),
        Path("/usr/local/opt/openjdk/bin"),
        *sorted(Path("/opt/homebrew/opt").glob("openjdk@*/bin")),
        *sorted(Path("/usr/local/opt").glob("openjdk@*/bin")),
    ]
    for candidate in candidates:
        if _javac_works(candidate):
            return candidate
    return None


class JavaBuildTask(Task):
    """Small Java source trees, each with its entry point and the output it must print."""

    name = "java_build"
    #: ``verify`` runs the pass's build.sh.
    graded_files = ("build.sh",)

    def items_root(self) -> Path:
        """Where this Task reads its items: the directory the package ships."""
        return ITEMS

    def instances(self) -> list[TaskInstance]:
        """One instance per source tree, in id order, with its entry point as metadata.

        A missing directory is refused by name: a broken install and an empty selection must not look alike.
        """
        root = self.items_root()
        if not root.is_dir():
            raise FileNotFoundError(
                f"the Java item directory {root} does not exist; the package ships it, "
                "so this is a broken install rather than an empty selection"
            )
        listed: list[TaskInstance] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            entry = json.loads((child / "item.json").read_text())
            listed.append(TaskInstance(
                id=child.name,
                path=str(child),
                metadata={
                    "main_class": str(entry.get("main_class") or ""),
                    "main_args": str(entry.get("main_args") or ""),
                },
            ))
        return listed

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        """Copy this item's sources into the run directory, replacing the previous item's."""
        source = Path(instance.path or "") / "src"
        for stale in ("src", "out"):
            shutil.rmtree(Path(workspace) / stale, ignore_errors=True)
        shutil.copytree(source, Path(workspace) / "src")

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Run the script the pass wrote and score it on what the program printed.

        One is what the whole pass is worth: the script compiled the tree and the entry point printed exactly the output recorded beside the sources. A missing script, a non-zero exit, a timeout, or different output are all zero, and the details carry what was observed so a reader can tell them apart.
        """
        workspace = Path(workspace)
        script = workspace / "build.sh"
        expected_path = Path(instance.path or "") / "expected.txt"
        expected = expected_path.read_text().strip() if expected_path.exists() else ""
        if not script.exists():
            return VerifyResult(
                passed=False, score=0.0, details="the pass wrote no build.sh"
            )
        env = dict(os.environ)
        toolchain = java_bin()
        if toolchain is not None:
            env["PATH"] = f"{toolchain}{os.pathsep}{env.get('PATH', '')}"
        try:
            completed = subprocess.run(
                ["sh", "build.sh"], cwd=str(workspace), capture_output=True,
                text=True, timeout=BUILD_TIMEOUT_S, env=env,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(
                passed=False, score=0.0,
                details=f"build.sh did not finish within {BUILD_TIMEOUT_S}s",
            )
        observed = completed.stdout.strip()
        passed = completed.returncode == 0 and observed == expected
        details = (
            f"exit {completed.returncode}; printed {observed!r}; expected {expected!r}; "
            f"stderr {completed.stderr.strip()[-400:]!r}"
        )
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0, details=details)
