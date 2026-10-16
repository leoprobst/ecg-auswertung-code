"""Race-tolerant memory sampling for an already-owned laboratory process."""
import os

import psutil


def live_tree_rss(process):
    """A vanished child is normal, not a reason to stop deadline monitoring.

    Access errors remain errors; unknown memory must not be called zero.
    This is a sampled total, not an atomic OS-enforced memory quota.
    """
    try:
        members = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def stop_owned_lab_tree(process, timeout=5):
    """Kill an owned worker and its currently discoverable descendants.

    psutil Process handles guard against PID reuse. This is not a Windows
    Job Object: descendants escaping before enumeration cannot be guaranteed.
    """
    if process.pid == os.getpid():
        raise ValueError("Refusing to stop the GUI/test host")
    errors = []
    try:
        descendants = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    except psutil.Error as exc:
        descendants = []
        errors.append(str(exc))
    members = [process, *descendants]
    # Stop the launcher first, then the captured descendants, retaining handles
    # even when their parent relationship disappears during termination.
    for member in members:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as exc:
            errors.append(str(exc))
    _, alive = psutil.wait_procs(members, timeout=timeout)
    if alive or errors:
        raise RuntimeError(
            f"Labor-Prozessbereinigung unvollstaendig: lebende PIDs "
            f"{[member.pid for member in alive]}; Fehler {errors}"
        )
