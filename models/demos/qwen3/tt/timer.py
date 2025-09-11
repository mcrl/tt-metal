import os
import ttnn
import time


def sync_and_time(device=None):
    if device is not None:
        ttnn.synchronize_device(device)
    else:
        ttnn.synchronize_device()
    return time.time()


TIMER = {}


def reset_timer():
    TIMER.clear()


def profile_enabled() -> bool:
    return os.getenv("PROFILE_TIME", "0") == "1"


def start_timer(key: str, device=None):
    if not profile_enabled():
        return
    now = sync_and_time(device)
    entry = TIMER.get(key)
    if entry is None:
        TIMER[key] = {"total": 0.0, "start": now, "count": 0, "samples": []}
    else:
        entry["start"] = now


def stop_timer(key: str, device=None):
    if not profile_enabled():
        return 0.0
    now = sync_and_time(device)
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
    if not profile_enabled():
        print("Profiling disabled (set PROFILE_TIME=1 to enable)")
        return
    for key, entry in TIMER.items():
        total = entry.get("total", 0.0)
        count = entry.get("count", 0)
        avg = (total / count) if count > 0 else 0.0
        samples = entry.get("samples", [])
        samples_str = ", ".join(f"{sample:.4f}s" for sample in samples)
        print(f"\n[{key}] total={total:.6f}s, count={count}, avg={avg:.6f}s")
        print(f"{samples_str}")


def timed(name: str | None = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not profile_enabled():
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
