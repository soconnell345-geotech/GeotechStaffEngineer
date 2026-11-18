"""Fixes from the review of the kernel layer (BACKLOG INF-16;
``module_work/inf16/REVIEW.md``). Every value is made up. The layer is
called directly, in a throwaway process, as in
``test_inf16_kernel_layer.py``.
"""

import errno
import json
import os
import socket
import subprocess
import sys

import pytest

from aec_agent_core.plot_box import kernel_layer

LAYER = os.path.abspath(kernel_layer.__file__)
LINUX = sys.platform.startswith("linux")

#: The file rights each Landlock version knows, as a count of bits from bit
#: 0 (the kernel's include/uapi/linux/landlock.h: 13 rights in version 1,
#: REFER with 2, TRUNCATE with 3, IOCTL_DEV with 5; version 4 added the TCP
#: rules and no file right).
FS_RIGHTS = {1: 13, 2: 14, 3: 15, 4: 15, 5: 16, 6: 16, 7: 16}


class _Recorder:
    """A stand-in C library: it keeps the rule set the layer asks the
    kernel for and answers "no" to everything, so nothing is put up."""

    def __init__(self):
        self.asked = None

    def syscall(self, number, *args):
        import ctypes
        if number.value == kernel_layer._NR_CREATE_RULESET:
            attr, size = args[0]._obj, args[1].value
            self.asked = (attr.handled_access_fs, attr.handled_access_net,
                          attr.scoped, size)
        ctypes.set_errno(errno.EINVAL)
        return -1


@pytest.mark.parametrize("version", sorted(FS_RIGHTS))
def test_each_landlock_version_is_asked_only_for_rights_it_knows(
        monkeypatch, version):
    """A kernel refuses (EINVAL) a rule set that names a right it does not
    know, and then no file rule goes up at all. Version 4 (Linux 6.7 to
    6.9) knows the TCP rules but not the right for device ioctl."""
    import ctypes
    monkeypatch.setattr(kernel_layer, "landlock_abi", lambda: (version, ""))
    libc = _Recorder()
    out = kernel_layer._landlock(ctypes, libc, [])
    assert not out["files"]                       # the stand-in said no
    fs, net, scoped, size = libc.asked
    assert fs == (1 << FS_RIGHTS[version]) - 1, hex(fs)
    assert net == (3 if version >= 4 else 0)
    assert scoped == (3 if version >= 6 else 0)
    assert size == (8 if version < 4 else 16 if version < 6 else 24)
