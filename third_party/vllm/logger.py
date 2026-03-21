"""Minimal vllm.logger shim for tpu-inference imports."""

import logging


class _VllmLogger(logging.Logger):
    pass


def init_logger(name: str) -> _VllmLogger:
    logger = logging.getLogger(name)
    logger.__class__ = _VllmLogger
    return logger
