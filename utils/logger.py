import logging
import os
import sys
import re


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO, write_file=True):
    logger = logging.getLogger(name)
    # remove all handlers
    for handler in logger.handlers[:]: #get rid of existing old handlers (with copy of logger)
      print('removing handler %s'%handler)
      logger.removeHandler(handler)

    # set logging level
    logger.setLevel(level)
    # Add file handler and stdout handler
    if write_file:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add console handler.
    console_handler = None
    for h in logging.getLogger().handlers[:]:
      if isinstance(h, logging.StreamHandler):
        console_handler = h
        logging.getLogger().removeHandler(console_handler)
        break
    if console_handler is None:
      console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    if write_file:
        logger.info('Log directory: %s', log_dir)
    else:
        logger.info('Console logging only')
    return logger


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def log_parser(log_dir, log_filename='info.log'):
    line_num = 0
    epoch_flag = re.compile(r'EVENT HEART')
    with open(os.path.join(log_dir, log_filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
            result_flag = epoch_flag.findall(line)
            if not result_flag:
                line_num += 1
                continue
