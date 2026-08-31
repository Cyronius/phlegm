"""Server configuration: paths, model id, and env overrides.

Everything here is a plain constant with an env override so the server can be
pointed at a different slice / driver / xclbin set without code edits. Nothing in
this module touches the NPU or numpy — importing it is free.
"""
import os

# --- backend selection -------------------------------------------------------
# "mock" (default, no device) | "npu" (drives the resident decode_driver).
# "auto" picks npu iff the driver + built buffers exist, else mock.
BACKEND = os.environ.get("FLM_BACKEND", "mock").lower()

# --- HTTP --------------------------------------------------------------------
HOST = os.environ.get("FLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLM_PORT", "52625"))  # FLM's own default port

# --- model identity (what /v1/models advertises) -----------------------------
MODEL_ID = os.environ.get("FLM_MODEL_ID", "qwen3.6-5li3-npu")

# --- generation defaults -----------------------------------------------------
DEFAULT_MAX_TOKENS = int(os.environ.get("FLM_MAX_TOKENS", "64"))
MAX_TOKENS_CAP = int(os.environ.get("FLM_MAX_TOKENS_CAP", "512"))

# --- NPU backend paths (only read when BACKEND resolves to npu) ---------------
REPO = os.environ.get("FLM_REPO", "C:/code/FastFlowLM")
NPU_OUT_DIR = os.environ.get("FLM_NPU_OUT_DIR", f"{REPO}/npu-engine/m3out/5li3")
DRIVER_EXE = os.environ.get("FLM_DRIVER_EXE", f"{REPO}/npu-engine/m0/out/decode_driver.exe")
XCLBIN_DIR = os.environ.get("FLM_XCLBIN_DIR", f"{REPO}/src/xclbins/Qwen3.6-35B-A3B-NPU2")
CAP_DIR = os.environ.get("FLM_CAP_DIR", "C:/caps/m0c")
KERNEL_INTERP_DIR = os.environ.get("FLM_KERNEL_INTERP_DIR", f"{REPO}/tools/kernel-interp")
MODEL_Q4NX = os.environ.get("FLM_MODEL_Q4NX", "model_5Li3.q4nx")
NUM_LAYERS = int(os.environ.get("FLM_NUM_LAYERS", "5"))

# Seconds to wait for the driver to reach "SERVE READY" (loads ~2.5GB of pools).
DRIVER_READY_TIMEOUT = float(os.environ.get("FLM_DRIVER_READY_TIMEOUT", "120"))
# Seconds to wait for a single decode step ("STEP OK").
DRIVER_STEP_TIMEOUT = float(os.environ.get("FLM_DRIVER_STEP_TIMEOUT", "30"))


def resolve_backend() -> str:
    """Turn 'auto' into a concrete backend name based on what's on disk."""
    if BACKEND != "auto":
        return BACKEND
    if os.path.exists(DRIVER_EXE) and os.path.exists(f"{NPU_OUT_DIR}/pool_lmhead.bin"):
        return "npu"
    return "mock"
