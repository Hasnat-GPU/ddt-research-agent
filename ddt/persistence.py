"""Exclusive local brain writes and recoverable publication of a group of files.

Readers of several files must hold locked() for a consistent view. A pending
journal restores the old state after interruption; a committed journal keeps
the new state. No unfinished state is used by a research run.
"""
import json
import os
import shutil
import threading
from contextlib import contextmanager

_mutex = threading.RLock()
_local = threading.local()


@contextmanager
def locked(root):
    root = root.resolve()
    with _mutex:
        held = getattr(_local, "held", {})
        if root in held:
            yield
            return
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a+b") as f:
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("brain is busy in another process; retry when that run finishes") from exc
            held[root] = True
            _local.held = held
            try:
                yield
            finally:
                del held[root]
                f.seek(0)
                if os.name == "nt":
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)


def _sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_bytes(path, data):
    temp = path.with_name(path.name + ".pending")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    _sync_dir(path.parent)


def copy_files(source, target, files):
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (source / name).exists():
            atomic_bytes(target / name, (source / name).read_bytes())
        else:
            (target / name).unlink(missing_ok=True)
    _sync_dir(target)


def _cleanup(root):
    work = root / ".transaction"
    if work.resolve().parent != root.resolve() or work.is_symlink():
        raise RuntimeError("unsafe transaction directory")
    if work.exists():
        shutil.rmtree(work)
        _sync_dir(root)


def recover(root, files):
    """Called only while holding the root lock, before reading committed state."""
    work = root / ".transaction"
    marker = work / "intent.json"
    if marker.exists():
        status = json.loads(marker.read_text(encoding="utf-8"))["status"]
        if status != "committed":
            copy_files(work / "before", root, files)
    _cleanup(root)


@contextmanager
def transaction(root, files):
    with locked(root):
        recover(root, files)
        work = root / ".transaction"
        before, after = work / "before", work / "after"
        copy_files(root, before, files)
        copy_files(root, after, files)
        atomic_bytes(work / "intent.json", b'{"status":"pending"}')
        try:
            yield after
            copy_files(after, root, files)
            atomic_bytes(work / "intent.json", b'{"status":"committed"}')
        except BaseException:
            # A recovery failure deliberately retains the journal for next startup.
            recover(root, files)
            raise
        _cleanup(root)
