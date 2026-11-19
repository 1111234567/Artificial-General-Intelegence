"""Compile CUDA C++ with PyTorch's bundled NVRTC and launch on its CUDA stream."""

from __future__ import annotations

import ctypes as C
import operator
from pathlib import Path
import re
import sys
from threading import Lock

import torch

_PTR = C.c_void_p
_PTR_OUT = C.POINTER(_PTR)
_STR_OUT = C.POINTER(C.c_char_p)
_SIZE_OUT = C.POINTER(C.c_size_t)
_INVALID_VALUE = 1  # CUDA_ERROR_INVALID_VALUE
_LOCK = Lock()
_LIBRARIES = None
# Graphs keep using the module after the Python launcher has returned. Keep
# loaded modules for the lifetime of their CUDA context (no cuModuleUnload).
_MODULES = {}


def _bind(library, result, functions):
    for name, arguments in functions.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = result


def _libraries():
    global _LIBRARIES
    if _LIBRARIES is None:
        if sys.platform != "win32":
            raise RuntimeError("the bundled NVRTC bridge requires Windows")
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        paths = [p for p in torch_lib.glob("nvrtc64_*.dll")
                 if re.fullmatch(r"nvrtc64_\d+_\d+\.dll", p.name)]
        if len(paths) != 1:
            raise RuntimeError(f"expected one NVRTC DLL in {torch_lib}; found {paths}")
        # Use the matching compiler from the Torch package. Torch has already
        # loaded its bundled DLLs (including the separately loaded builtins).
        nvrtc = C.CDLL(str(paths[0]))
        driver = C.WinDLL("nvcuda.dll", winmode=0x00000800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        _bind(nvrtc, C.c_char_p, {"nvrtcGetErrorString": (C.c_int,)})
        _bind(nvrtc, C.c_int, {
            "nvrtcCreateProgram": (
                _PTR_OUT, C.c_char_p, C.c_char_p, C.c_int, _STR_OUT, _STR_OUT,
            ),
            "nvrtcCompileProgram": (_PTR, C.c_int, _STR_OUT),
            "nvrtcGetProgramLogSize": (_PTR, _SIZE_OUT),
            "nvrtcGetProgramLog": (_PTR, _PTR),
            "nvrtcGetCUBINSize": (_PTR, _SIZE_OUT),
            "nvrtcGetCUBIN": (_PTR, _PTR),
            "nvrtcDestroyProgram": (_PTR_OUT,),
        })
        _bind(driver, C.c_int, {
            "cuInit": (C.c_uint,),
            "cuGetErrorName": (C.c_int, _STR_OUT),
            "cuGetErrorString": (C.c_int, _STR_OUT),
            "cuCtxGetCurrent": (_PTR_OUT,),
            "cuCtxGetDevice": (C.POINTER(C.c_int),),
            "cuModuleLoadData": (_PTR_OUT, _PTR),
            "cuModuleGetFunction": (_PTR_OUT, _PTR, C.c_char_p),
            "cuFuncGetParamInfo": (_PTR, C.c_size_t, _SIZE_OUT, _SIZE_OUT),
            "cuLaunchKernel": (_PTR, *([C.c_uint] * 7), _PTR, _PTR_OUT, _PTR_OUT),
        })
        if hasattr(driver, "cuFuncLoad"):
            _bind(driver, C.c_int, {"cuFuncLoad": (_PTR,)})
        _cuda_check(driver, "cuInit", driver.cuInit(0))
        _LIBRARIES = driver, nvrtc
    return _LIBRARIES


def _decode(value):
    return (value or b"").decode("utf-8", errors="replace")


def _cuda_check(driver, operation, result):
    if result:
        name, detail = C.c_char_p(), C.c_char_p()
        driver.cuGetErrorName(result, C.byref(name))
        driver.cuGetErrorString(result, C.byref(detail))
        raise RuntimeError(f"{operation}: {_decode(name.value)} ({result}): {_decode(detail.value)}")


def _nvrtc_check(nvrtc, operation, result, log=""):
    if result:
        name = _decode(nvrtc.nvrtcGetErrorString(result))
        raise RuntimeError(f"{operation}: {name} ({result})" + (f"\n{log}" if log else ""))


def _compile(nvrtc, source, options):
    program = _PTR()
    status = nvrtc.nvrtcCreateProgram(C.byref(program), source, b"kernels.cu", 0, None, None)
    _nvrtc_check(nvrtc, "nvrtcCreateProgram", status)
    try:
        argv = (C.c_char_p * len(options))(*options)
        status = nvrtc.nvrtcCompileProgram(program, len(argv), argv)
        if status:
            size = C.c_size_t()
            log = ""
            if nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)) == 0 and size.value:
                buf = C.create_string_buffer(size.value)
                if nvrtc.nvrtcGetProgramLog(program, buf) == 0:
                    log = _decode(buf.value).rstrip()
            _nvrtc_check(nvrtc, "nvrtcCompileProgram", status, log)
        size = C.c_size_t()
        _nvrtc_check(nvrtc, "nvrtcGetCUBINSize", nvrtc.nvrtcGetCUBINSize(program, C.byref(size)))
        if not size.value:
            raise RuntimeError("NVRTC returned no CUBIN for the requested architecture")
        image = C.create_string_buffer(size.value)
        _nvrtc_check(nvrtc, "nvrtcGetCUBIN", nvrtc.nvrtcGetCUBIN(program, image))
    finally:
        status = nvrtc.nvrtcDestroyProgram(C.byref(program))
    _nvrtc_check(nvrtc, "nvrtcDestroyProgram", status)
    return image


def _current_context(driver, device):
    context, ordinal = _PTR(), C.c_int()

    def query():
        _cuda_check(driver, "cuCtxGetCurrent", driver.cuCtxGetCurrent(C.byref(context)))
        if context.value:
            _cuda_check(driver, "cuCtxGetDevice", driver.cuCtxGetDevice(C.byref(ordinal)))
            return ordinal.value == device.index
        return False

    if not query():
        # Establish Torch's primary context if no CUDA tensor has been
        # allocated on this device yet. This runs only before capture.
        torch.empty(1, device=device)
        if not query():
            raise RuntimeError(f"Torch has no current CUDA context for {device}")
    return context.value


def _dimensions(dims):
    try:
        dims = (operator.index(dims),)
    except TypeError:
        dims = tuple(map(operator.index, dims))
    if not 1 <= len(dims) <= 3 or any(not 0 < d <= 0xFFFFFFFF for d in dims):
        raise ValueError("CUDA grid and block each require one to three positive dimensions")
    return dims + (1,) * (3 - len(dims))


def _argument(value, device):
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError(f"kernel tensor argument is on {value.device}, expected {device}")
        return _PTR(value.data_ptr())
    if isinstance(value, int):
        if not -(1 << 31) <= value < (1 << 31):
            raise OverflowError("use an explicit ctypes type for a CUDA integer outside int32")
        return C.c_int32(value)
    if isinstance(value, float):
        return C.c_float(value)
    return value  # Explicit ctypes values provide their own storage and width.


def _launcher(driver, module, device, name):
    function = _PTR()
    _cuda_check(driver, f"cuModuleGetFunction({name})",
                driver.cuModuleGetFunction(C.byref(function), module, name.encode("ascii")))
    if hasattr(driver, "cuFuncLoad"):
        _cuda_check(driver, f"cuFuncLoad({name})", driver.cuFuncLoad(function))
    # The driver copies parameters using the CUBIN's ABI and receives no
    # count. Catch short argument arrays or undersized values before launch.
    sizes = []
    offset, size = C.c_size_t(), C.c_size_t()
    while True:
        result = driver.cuFuncGetParamInfo(function, len(sizes), C.byref(offset), C.byref(size))
        if result == _INVALID_VALUE:
            break
        _cuda_check(driver, f"cuFuncGetParamInfo({name})", result)
        sizes.append(size.value)

    def launch(grid, block, *params, shared_mem=0):
        if len(params) != len(sizes):
            raise TypeError(f"{name} expects {len(sizes)} CUDA arguments; got {len(params)}")
        stream = torch.cuda.current_stream()
        if stream.device != device:
            raise RuntimeError(f"{name} is loaded on {device}; current stream is on {stream.device}")
        grid, block = _dimensions(grid), _dimensions(block)
        shared_mem = operator.index(shared_mem)
        if not 0 <= shared_mem <= 0xFFFFFFFF:
            raise ValueError("shared_mem must be a nonnegative uint32 byte count")
        # Keep the scalar/pointer storage live until cuLaunchKernel copies it.
        values = [_argument(p, device) for p in params]
        addresses = (_PTR * len(values))()
        for i, (value, expected_size) in enumerate(zip(values, sizes)):
            try:
                address, actual_size = C.addressof(value), C.sizeof(value)
            except TypeError as error:
                raise TypeError(f"{name} argument {i + 1} is not a tensor, int, float, or ctypes value") from error
            if actual_size != expected_size:
                raise TypeError(f"{name} argument {i + 1} needs {expected_size} bytes; got {actual_size}")
            addresses[i] = address
        result = driver.cuLaunchKernel(
            function, *grid, *block, shared_mem, stream.cuda_stream,
            addresses if values else None, None,
        )
        _cuda_check(driver, f"cuLaunchKernel({name})", result)

    return launch


def compile_kernels(source, names, device=None, options=()):
    """Return launchers for named ``extern "C"`` kernels, compiled before capture.

    Call a launcher as ``kernel(grid, block, *args, shared_mem=0)`` with the
    compiled device active. Tensors pass device pointers, Python ints/floats
    pass int32/float32, and ctypes values can supply other ABI types. As for
    other asynchronous CUDA work, callers manage cross-stream ordering and
    buffer lifetimes, including keeping graph input/output buffers for replay.
    """
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("NVRTC kernels require a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    names = (names,) if isinstance(names, str) else tuple(names)
    if isinstance(options, str):
        raise TypeError("options must be a sequence of NVRTC option strings")
    options = tuple(options)
    flags = {option.split("=", 1)[0] for option in options}
    if flags & {"--gpu-architecture", "-arch"}:
        raise ValueError("the NVRTC architecture is selected from the CUDA device")
    if not flags & {"--fmad", "-fmad", "--use_fast_math", "-use_fast_math"}:
        options = ("--fmad=false", *options)
    source = source.encode("utf-8")
    with _LOCK, torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile_kernels must be called before CUDA graph capture")
        driver, nvrtc = _libraries()
        context = _current_context(driver, device)
        major, minor = torch.cuda.get_device_capability(device)
        options = tuple(s.encode("utf-8") for s in (f"--gpu-architecture=sm_{major}{minor}", *options))
        key = (context, source, options)
        if key not in _MODULES:
            image = _compile(nvrtc, source, options)
            module = _PTR()
            _cuda_check(driver, "cuModuleLoadData", driver.cuModuleLoadData(C.byref(module), image))
            _MODULES[key] = module, {}
        module, launchers = _MODULES[key]
        for name in names:
            if name not in launchers:
                launchers[name] = _launcher(driver, module, device, name)
        return {name: launchers[name] for name in names}
