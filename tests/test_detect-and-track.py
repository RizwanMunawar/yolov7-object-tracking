import subprocess
import sys


def run_script(script, args):
    """Helper to run a script with arguments and return the exit code."""
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print output to help debugging in CI
    print(result.stdout)
    print(result.stderr)
    return result.returncode


def test_detect_py():
    exit_code = run_script(
        "detect.py",
        ["--weights", "yolov7.pt", "--conf-thres", "0.25", "--nosave"],
    )
    assert exit_code == 0, "detect.py failed to run successfully"


def test_detect_and_track_py():
    exit_code = run_script(
        "detect_and_track.py",
        ["--weights", "yolov7.pt", "--conf-thres", "0.25", "--nosave"],
    )
    assert exit_code == 0, "detect_and_track.py failed to run successfully"
