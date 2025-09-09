"""
Minimal psutil shim for tests that only need memory_info().
This avoids adding a heavy dependency in constrained environments.
"""
from dataclasses import dataclass

@dataclass
class _MemInfo:
    rss: int = 150 * 1024 * 1024  # 150 MB baseline

class Process:
    def __init__(self, pid: int):
        self._pid = pid
    def memory_info(self):
        return _MemInfo()
