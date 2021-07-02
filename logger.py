from loguru import logger
import sys

def setup_logger(filter=None, level="INFO"):
    """ Setup logger
    
    Args:
        filter (str): Filter to apply, e.g. 'blocks.read_vcs'
        level (str): Logging level, e.g. INFO, DEBUG, WARNING
    """
    config = {
        "handlers": [
            {"sink": sys.stdout, 
             "format": "<green>[{time}]</green> <blue>{level}</blue> {message}", 
             "filter": filter, 
             "level": level,
             "colorize": True
            },
        ],
    }    
    logger.configure(**config)