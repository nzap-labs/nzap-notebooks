# NZAP Engine injects a `params` dict before this script runs.
# This notebook takes no parameters.

import os
import platform
import shutil
import sys

print(f"Python {sys.version.split()[0]} on {platform.platform()}")
print(f"Machine: {platform.machine()}, {os.cpu_count()} CPUs")
total, used, free = shutil.disk_usage("/")
print(f"Disk: {used / 1e9:.1f} GB used of {total / 1e9:.1f} GB")
try:
    with open("/proc/meminfo") as handle:
        fields = dict(line.split(":", 1) for line in handle)
    print(f"RAM: {fields['MemTotal'].strip()} total, {fields['MemAvailable'].strip()} available")
except OSError:
    pass
print(f"params received: {params}")
