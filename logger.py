import logging
import os

#logging.basicConfig(level=logging.INFO)

def setup_logger(
    name="app",
    log_file="logs/app.log",
    level=logging.INFO
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Log to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Log to terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger