import datetime

def log_message(log, message, to_console=True):
    """
    Write a log line to file, optionally also print to console.

    Args:
        log: file handle for writing.
        message: log content string.
        to_console: whether to mirror to stdout.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    log.write(formatted_message + "\n")
    log.flush()
    if to_console:
        print(formatted_message)