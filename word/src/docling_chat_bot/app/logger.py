import logging

# Creating a named logger
logger = logging.getLogger("docling_api")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Console handler for outputting logs to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define log format
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if this module is reimported
if not logger.hasHandlers():
    logger.addHandler(console_handler)
