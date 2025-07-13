import sys
import UFSR.LogFile

def error_exit_1(log_output: str) -> None:
    global_logger = UFSR.LogFile.GlobalLogger1()
    log = global_logger.get_logger_1()
    log.error(log_output)
    sys.exit()

def error_warning_1(log_output: str) -> None:
    global_logger = UFSR.LogFile.GlobalLogger1()
    log = global_logger.get_logger_1()
    log.warning(log_output)