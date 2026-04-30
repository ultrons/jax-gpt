# Bootstrap: mock vllm (not installed here) and add tpu-inference to path.
# Import this module first in every script:  import env  # noqa: F401
import sys
import types
import logging

_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")

# Subclass Logger to add vllm-specific methods used by tpu_inference code.
class _VllmLogger(logging.Logger):
    def warning_once(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)
    def info_once(self, msg, *args, **kwargs):
        self.info(msg, *args, **kwargs)
    def debug_once(self, msg, *args, **kwargs):
        self.debug(msg, *args, **kwargs)

logging.setLoggerClass(_VllmLogger)

def _init_logger(name):
    return logging.getLogger(name)

_vllm_logger.init_logger = _init_logger
_vllm_logger._VllmLogger = _VllmLogger
_vllm_logger.init_vllm_logger = _init_logger
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.logger", _vllm_logger)

_TPU_INFERENCE = "/home/sivaibhav_google_com/tpu-inference"
if _TPU_INFERENCE not in sys.path:
    sys.path.insert(0, _TPU_INFERENCE)
