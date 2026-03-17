from __future__ import annotations
from pathlib import Path # 文件路径
import logging
def configure_console_logging(
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s", # Log的格式
    datefmt: str = "%H:%M:%S",
) -> None:
    '''
    Logging utility, set up console logging for scripts.
    
    Library code should NOT call this automatically.
    Only call it from entry points (experiments, scripts, CLI).
    '''
    
    root = logging.getLogger() # returns root logger
    if root.handlers:
        # Avoid double logging if configure_logging is called multiple times
        root.setLevel(level)
        return

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt) # logging.basicConfig会自动添加一个console handler，因此需要上方的if。
    
def add_file_handlers(
    *,
    log_dir: str = "logs",
    all_log_name: str = "app.log",
    err_log_name: str = "error.log",
    level_all: int = logging.DEBUG,
    level_err: int = logging.ERROR,
    fmt: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    
    '''Add file handlers to the root logger.

    - all_log_name receives everything >= level_all
    - err_log_name receives everything >= level_err'''
    
    root = logging.getLogger()

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    all_path = Path(log_dir) / all_log_name
    err_path = Path(log_dir) / err_log_name

    # Avoid duplicates if called multiple times
    existing = {getattr(h, "baseFilename", None) for h in root.handlers}

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if str(all_path) not in existing:
        fh_all = logging.FileHandler(all_path, encoding="utf-8")
        fh_all.setLevel(level_all)
        fh_all.setFormatter(formatter)
        root.addHandler(fh_all)

    if str(err_path) not in existing:
        fh_err = logging.FileHandler(err_path, encoding="utf-8")
        fh_err.setLevel(level_err)
        fh_err.setFormatter(formatter)
        root.addHandler(fh_err)


'''
Loggers是分层的对象，可以释放log信息，handler决定log信息去哪里（handler只应该存在在应用入口，而不应该在library code中）。Logger的等级包括：
DEBUG    – Diagnostic detail
INFO     – Normal operation
WARNING  – Unexpected but recoverable
ERROR    – Operation failed
CRITICAL – System is unusable
Pythong中对应一个固定的整数。

logging.info()等函数创造log信息，logging.basicConfig()决定最低等级的log信息，以上的会被输出看到。
'''