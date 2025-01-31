import logging
import inspect
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        level_name = record.levelname
        color = self.COLORS[level_name]


        # Create the colored level name in brackets
        time = "%(asctime)s"
        color_level = f"{color}%(levelname)-8s{self.RESET}"
        log_name = "[%(name)s]"
        message = "%(message)s"
        line_number = "%(lineno)s"
        filename = "%(filename)s"

        log_format = f"{time} {log_name} {color_level}"

        # Add the linenumber only if its a debug log
        if level_name == "DEBUG":
            log_format += f"{filename} {line_number}"

        # Add the message 
        log_format += f": {message}"

        time_format = "%H:%M:%S"

        formatter = logging.Formatter(log_format, time_format)
        return formatter.format(record)


def setup_logger(default_log_level = logging.DEBUG):
    adjust_all_loggers()
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(default_log_level)

    # Set up a console handler with the custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())

    # Add the handler to the logger
    logger.addHandler(console_handler)

def adjust_all_loggers(default_level=logging.ERROR):
    # Get all defined loggers
    all_loggers = logging.root.manager.loggerDict

    # Loop through all loggers and set their levels
    for logger_name, logger_obj in all_loggers.items():
        # Ensure it's a logger
        if isinstance(logger_obj, logging.Logger):
            # Ignore shap loggers
            if logger_name.startswith("shap"):
                continue

            # Set the level to default
            logger_obj.setLevel(default_level)

if __name__ == "__main__":
    setup_logger()

    logger = logging.getLogger(__name__)
    # Example usage
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
