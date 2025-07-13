import logging
import sys

def get_logger_1(log_name: str, console_output: bool = False, file_output: bool = False, \
                 console_level: str = "WARNING", file_handler: dict[str,list[str]] = {}) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(get_level_1(console_level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if file_output:
        for handler in file_handler:
            fh = logging.FileHandler(file_handler[handler][0], file_handler[handler][1])
            fh.setLevel(get_level_1(file_handler[handler][2]))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def get_logger_2(log_name: str, console_output: bool = False, file_output: bool = False, \
                 log_formatter: str = "%(asctime)s - %(levelname)s - %(message)s", \
                 console_level: str = "WARNING", file_handler: dict[str,list[str]] = {}) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_formatter)
    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(get_level_1(console_level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if file_output:
        for handler in file_handler:
            fh = logging.FileHandler(file_handler[handler][0], file_handler[handler][1])
            fh.setLevel(get_level_1(file_handler[handler][2]))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def get_level_1(level_name: str):
    levels = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    if level_name in levels:
        return levels[level_name]
    else:
        print(f"LOG Level Name Error: {level_name}")
        sys.exit()

class GlobalLogger1:
    _instance = None
    _logger_configured = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalLogger1, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_name: str = "default", console_output: bool = False, file_output: bool = False, 
                 log_formatter: str = "%(asctime)s - %(levelname)s - %(message)s", 
                 console_level: str = "WARNING", file_handler: dict[str, list[str]] = {}):
        if not self._logger_configured:
            self.logger = logging.getLogger(log_name)
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(log_formatter)
            if console_output:
                ch = logging.StreamHandler()
                ch.setLevel(self.get_level_1(console_level))
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            if file_output:
                for handler in file_handler:
                    fh = logging.FileHandler(file_handler[handler][0], file_handler[handler][1])
                    fh.setLevel(self.get_level_1(file_handler[handler][2]))
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)
            self._logger_configured = True

    def get_logger_1(self):
        return self.logger

    def get_level_1(self, level_name: str):
        levels = {
            "NOTSET": logging.NOTSET,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        if level_name in levels:
            return levels[level_name]
        else:
            print(f"LOG Level Name Error: {level_name}")
            sys.exit()
