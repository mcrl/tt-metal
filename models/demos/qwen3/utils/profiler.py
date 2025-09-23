import os
import json
import time
import threading
import atexit
import fcntl
from functools import wraps
from models.demos.qwen3.utils.timer import start_timer, stop_timer, timer_enabled, set_and_get_device_cache

CATEGORIES = {0: "Phase", 1: "Program", 2: "Layer", 3: "Block", 4: "Operation"}
EVENTS = []
LOCK = threading.RLock()
ENABLED = True

def profiler_enabled() -> bool:
    return os.getenv("PROFILE_TRACE", "0") == "1"

def level_enabled(level: int) -> bool:
    level_list = os.getenv("PROFILE_TRACE_LEVELS", "0,1,2,3,4")
    enabled_levels = set(int(lvl) for lvl in level_list.split(","))
    return level in enabled_levels

def get_trace_file() -> str:
    return os.getenv("PROFILE_TRACE_FILENAME", "trace.json")

def init_trace_file():
    output_file = get_trace_file()

    if not os.path.exists(output_file):
        return
    
    name, ext = os.path.splitext(output_file)

    prefix = 1
    while True:
        new_filename = f"{name}_{prefix}{ext}"
        if not os.path.exists(new_filename):
            os.environ["PROFILE_TRACE_FILENAME"] = new_filename
            break
        prefix += 1

def save_process_events_on_exit():
    if profiler_enabled():
        output_file = get_trace_file()
        try:
            with LOCK:
                current_events = EVENTS.copy()

            with open(output_file, 'a+') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except:
                    pass
                
                f.seek(0)
                existing_events = []
                try:
                    content = f.read()
                    if content.strip():
                        existing_data = json.loads(content)
                        if isinstance(existing_data, dict) and 'traceEvents' in existing_data:
                            existing_events = existing_data['traceEvents']
                        elif isinstance(existing_data, list):
                            existing_events = existing_data
                except Exception as e:
                    print(f"WARNING: Failed to read existing {output_file}: {e}")
                
                all_events = existing_events + current_events
                all_events.sort(key=lambda x: x.get('ts', 0))
                
                trace_data = {
                    "traceEvents": all_events,
                    "displayTimeUnit": "ms"
                }
                
                f.seek(0)
                f.truncate()
                json.dump(trace_data, f, indent=2)
                        
        except Exception as e:
            print(f"Failed to save events: {e}")

if profiler_enabled():
    atexit.register(save_process_events_on_exit)

def enable_profiler():
    global ENABLED
    ENABLED = True

def disable_profiler():
    global ENABLED
    ENABLED = False

def add_event(event):
    if not profiler_enabled() or not ENABLED:
        return
    
    try:
        with LOCK:
            EVENTS.append(event)
    except Exception as e:
        print(f"Failed to add event: {e}")

def get_events():
    try:
        with LOCK:
            return EVENTS.copy()
    except Exception as e:
        print(f"Failed to get events: {e}")
        return []   

def clear_events():
    global EVENTS
    try:
        with LOCK:
            EVENTS = []
    except Exception as e:
        print(f"Failed to clear events: {e}")


class ProfilerTimerContext:
    def __init__(self, name, level, args=None):
        self.name = name
        self.level = level
        self.args = args or {}
        self.device = set_and_get_device_cache()
        self.start_time = None
        
    def __enter__(self):
        if profiler_enabled():
            self.start_time = time.perf_counter() * 1e6
        
        if timer_enabled():
            start_timer(self.name, device=self.device)
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = None
        if timer_enabled():
            elapsed_time = stop_timer(self.name, device=self.device)

        if profiler_enabled() and self.start_time is not None and level_enabled(self.level):
            end_time = time.perf_counter() * 1e6
            tid = self.level
            pid = os.getpid()
            
            begin_event = {
                "name": self.name,
                "cat": CATEGORIES.get(self.level, "Unknown"),
                "ph": "B",
                "pid": pid,
                "tid": tid,
                "ts": self.start_time
            }
            if self.args:
                begin_event["args"] = self.args
            
            end_event = {
                "name": self.name,
                "cat": CATEGORIES.get(self.level, "Unknown"),
                "ph": "E",
                "pid": pid,
                "tid": tid,
                "ts": end_time
            }
            if self.args:
                end_event["args"] = self.args

            add_event(begin_event)
            add_event(end_event)


class Profiler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._open_events = {}
            self._pid = os.getpid()
            self._initialized = True

            if profiler_enabled() and len(EVENTS) == 0:
                initial_event = {"name": "process_name", "ph": "M", "pid": self._pid, "args": {"name": f"Process {self._pid}"}}
                add_event(initial_event)

    @staticmethod
    def _create_event(name, cat, ph, pid, tid, ts, args=None):
        event = {
            "name": name,
            "cat": cat,
            "ph": ph,
            "pid": pid,
            "tid": tid,
            "ts": ts,
        }
        if args:
            event["args"] = args
        return event

    def __enter__(self):
        if not profiler_enabled():
            return self
        
        self.start_time = time.perf_counter() * 1e6
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not profiler_enabled():
            return
        
        if not hasattr(self, '_open_events') or not self._open_events:
            return
        
        end_time = time.perf_counter() * 1e6
        
        name = self._open_events.get("name")
        level = self._open_events.get("level")
        args = self._open_events.get("args")
        
        if name is not None and level is not None and level_enabled(level):
            tid = level
            
            begin_event = self._create_event(
                name, CATEGORIES.get(level, "Unknown"), "B", self._pid, tid, self.start_time, args)
            end_event = self._create_event(
                name, CATEGORIES.get(level, "Unknown"), "E", self._pid, tid, end_time, args)
            
            add_event(begin_event)
            add_event(end_event)
        
        self._open_events = {}

    def trace(self, name, level, args=None):
        if not profiler_enabled():
            return self
        self._open_events = {"name": name, "level": level, "args": args}
        return self

    def trace_with_timer(self, name, level, args=None):
        return ProfilerTimerContext(name, level, args)

    @classmethod
    def reset(cls):
        clear_events()
        
        if cls._instance:
            cls._instance._open_events = {}


def profile_trace(name=None, level=1, args=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*func_args, **func_kwargs):
            trace_name = name if name is not None else func.__name__
            
            with ProfilerTimerContext(trace_name, level, args):
                return func(*func_args, **func_kwargs)
        
        return wrapper
    
    return decorator