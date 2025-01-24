import logging

def setup_logger(log_file="app.log", log_level=logging.INFO):
    """
    Sets up a logger with a specified log file and log level.
    
    Parameters:
        log_file (str): Path to the log file (default is "app.log").
        log_level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Create a logger instance
    logger = logging.getLogger("SimpleLogger")
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Stream handler to display logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Set the log message format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger