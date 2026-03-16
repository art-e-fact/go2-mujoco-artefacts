import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from typing import Any, List

import psutil

logger = logging.getLogger(__name__)


@contextmanager
def ignore_interupt():
    """Codeblock in this context will not be affected by SIGINT"""
    original_handler = signal.getsignal(signal.SIGINT)
    inter = False

    def cbk(*_, **__):
        nonlocal inter
        inter = True

    try:
        signal.signal(signal.SIGINT, cbk)
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if inter:
            raise KeyboardInterrupt


def finish_process(
    p: subprocess.Popen,
    timeout: float = 2.0,
    repetition: int = 3,
    kill_timeout: float = 1.0,
    kill_repetition: int = 3,
):
    """
    Terminates a subprocess.Popen process and its children.

    Args:
        p: subprocess.Popen object
        timeout: seconds to wait SIGTERM before escalating to SIGKILL
        repetition: number of SIGTERM to send before timeout
        kill_timeout: seconds to wait after SIGKILL before giving up
        kill_repetition: number of SIGKILL to send before timeout
    """
    # p.poll() and such does not work because it doesn't consider the childs

    timeout = timeout / repetition
    kill_timeout = kill_timeout / kill_repetition
    try:
        parent_proc = psutil.Process(p.pid)
    except psutil.NoSuchProcess:
        print("SHUDOWN: already dead (proc)")
        return True

    all_proc = parent_proc.children(recursive=True) + [parent_proc]

    def is_stopped() -> bool:
        nonlocal all_proc
        for proc in all_proc:
            is_running = proc.is_running()
            is_in_reality_dead = False
            if is_running:
                is_in_reality_dead = proc.status() in [
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                ]
            is_alive = is_running and not is_in_reality_dead
            if is_alive:
                # print(f"still running: {proc.name}")
                return False
        return True

    for _ in range(repetition):
        # send SIGTERM to all childs
        logger.debug("Sending: SIGTERM")
        for proc in all_proc:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # wait for graceful exit
        deadline = time.time() + timeout
        while time.time() < deadline:
            if is_stopped():
                logger.debug("SUCCESS: on SIGTERM")
                return
            time.sleep(0.1)

    for _ in range(kill_repetition):
        # send SIGKILL to all childs
        logger.debug("Sending: SIGKILL")
        for proc in all_proc:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # wait for graceful exit
        deadline = time.time() + kill_timeout
        while time.time() < deadline:
            if is_stopped():
                logger.debug("SUCCESS: on SIGKILL")
                return
            time.sleep(0.1)
