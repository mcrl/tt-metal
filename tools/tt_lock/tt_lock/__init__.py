import fcntl
import os
import sys
import atexit
import time

LOCK_FILE = "/tmp/tt_lock.lock"
_lock_fd = None

def _cleanup_lock():
    """
    Clean up function to release lock and remove lock file.
    """
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)  # Explicitly unlock
            os.close(_lock_fd)
        except:
            pass
        finally:
            _lock_fd = None
    
    # Remove lock file when releasing
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            print("Lock released.")
    except:
        pass

def acquire_lock():
    """
    Acquires an exclusive file lock, blocking until successful.
    """
    global _lock_fd
    
    # Clean up any existing lock first
    if _lock_fd is not None:
        _cleanup_lock()
    
    while True:
        try:
            _lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("Successfully acquired lock.")
            atexit.register(_cleanup_lock)
            return
        except BlockingIOError:
            # Close the file descriptor before waiting
            if _lock_fd is not None:
                os.close(_lock_fd)
                _lock_fd = None
            print("Another process holds the lock. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            if _lock_fd is not None:
                os.close(_lock_fd)
                _lock_fd = None
            sys.exit(1)

acquire_lock()