import os
import ttnn
import time

TIMER = {}
DEVICE_CACHE = None


def reset_timer():
    TIMER.clear()


def timer_enabled() -> bool:
    return os.getenv("PROFILE_TIME", "0") == "1"


def set_and_get_device_cache(device=None):
    global DEVICE_CACHE
    if device is not None:
        DEVICE_CACHE = device
    else:
        if DEVICE_CACHE is None:
            raise ValueError("Device is not set")
    return DEVICE_CACHE


def start_timer(key: str, device=None):
    if not timer_enabled():
        return

    device = set_and_get_device_cache(device)

    now = time.time()
    entry = TIMER.get(key)
    if entry is None:
        TIMER[key] = {"total": 0.0, "start": now, "count": 0, "samples": []}
    else:
        entry["start"] = now


def stop_timer(key: str, device=None):
    if not timer_enabled():
        return 0.0

    device = set_and_get_device_cache(device)

    now = time.time()
    entry = TIMER.get(key)
    if entry is None:
        return 0.0
    start = entry.get("start")
    if start is None:
        return 0.0
    elapsed = now - start
    entry["total"] += elapsed
    entry["count"] += 1
    entry.setdefault("samples", []).append(elapsed)
    entry["start"] = None
    return elapsed


def get_timer(key: str):
    entry = TIMER.get(key)
    if entry is None:
        return {"total": 0.0, "count": 0, "samples": []}
    return {"total": entry.get("total", 0.0), "count": entry.get("count", 0), "samples": list(entry.get("samples", []))}


def print_timer_all():
    if not timer_enabled():
        print("Profiling disabled (set PROFILE_TIME=1 to enable)")
        return
    for key, entry in TIMER.items():
        total = entry.get("total", 0.0)
        count = entry.get("count", 0)
        avg = (total / count) if count > 0 else 0.0
        min_sample = min(entry.get("samples", []))
        max_sample = max(entry.get("samples", []))
        samples = entry.get("samples", [])
        samples_str = ", ".join(f"{sample * 1000:.2f}ms" for sample in samples)
        print(f"\n[{key}] total={total * 1000:.3f}ms, count={count}, avg={avg * 1000:.3f}ms, min={min_sample * 1000:.2f}ms, max={max_sample * 1000:.2f}ms")
        print(f"{samples_str}")


def timed(name: str | None = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not timer_enabled():
                return func(*args, **kwargs)
            self_obj = args[0] if args else None
            device = getattr(self_obj, "mesh_device", None)
            key = name or (self_obj.__class__.__name__ if self_obj is not None else func.__name__)
            start_timer(key, device=device)
            try:
                return func(*args, **kwargs)
            finally:
                stop_timer(key, device=device)
        return wrapper
    return decorator


def profile_time(name: str | None = None):
    return timed(name)
