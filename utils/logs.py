import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    __logger = logging.getLogger()
    __logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

    __logger.addHandler(console_handler)

    @staticmethod
    def set_log_level(level):
        """
        Sets the log level for the logger. The level parameter determines the verbosity of the logs.
        Args:
            level:

        Returns:

        """
        logging.getLogger().setLevel(level)

    @staticmethod
    def info(content):
        """
        Logs an informational message with the provided content.
        Args:
            content:

        Returns:

        """
        Logger.__logger.info(content, stacklevel=2)

    @staticmethod
    def debug(content):
        """
        Logs a debug message with the provided content.
        Args:
            content:

        Returns:

        """
        Logger.__logger.debug(content, stacklevel=2)

    @staticmethod
    def error(content):
        """
        Logs an error message with the provided content.
        Args:
            content:

        Returns:

        """
        Logger.__logger.error(content, stacklevel=2)

    @staticmethod
    def critical(content):
        """
        Logs a critical message with the provided content.
        Args:
            content:

        Returns:

        """
        Logger.__logger.critical(content, stacklevel=2)

    @staticmethod
    def warning(content):
        """
        Logs a warning message with the provided content.
        Args:
            content:

        Returns:

        """
        Logger.__logger.warning(content, stacklevel=2)


if __name__ == '__main__':
    Logger.warning('What are you doing now?')