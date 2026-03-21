"""Minimal vllm.logger shim for tpu-inference imports.

tpu-inference requires vllm.logger at import time. This shim provides
the minimal interface so we can use the RPA kernel without installing
the full vllm package.

Usage: import this module before importing tpu_inference, or set
PYTHONPATH so that this file's parent is found as 'vllm'.
"""

import logging


class _VllmLogger(logging.Logger):
    pass


def init_logger(name: str) -> _VllmLogger:
    logger = logging.getLogger(name)
    logger.__class__ = _VllmLogger
    return logger
