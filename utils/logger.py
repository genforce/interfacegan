# python3.7
"""Utility functions for logging."""

import os
import sys
import logging

__all__ = ['setup_logger']


def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
  """Sets up logger from target work directory.

  The function will sets up a logger with `DEBUG` log level. Two handlers will
  be added to the logger automatically. One is the `sys.stdout` stream, with
  `INFO` log level, which will print improtant messages on the screen. The other
  is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
  be added time stamp and log level before logged.

  NOTE: If `work_dir` or `logfile_name` is empty, the file stream will be
  skipped.

  Args:
    work_dir: The work directory. All intermediate files will be saved here.
      (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

  Returns:
    A `logging.Logger` object.

  Raises:
    SystemExit: If the work directory has already existed, of the logger with
      specified name `logger_name` has already existed.
  """

  logger = logging.getLogger(logger_name)
  if logger.hasHandlers():  # Already existed
    raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                     f'Please use another name, or otherwise the messages '
                     f'may be mixed between these two loggers.')

  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

  # Print log message with `INFO` level or above onto the screen.
  sh = logging.StreamHandler(stream=sys.stdout)
  sh.setLevel(logging.INFO)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  if not work_dir or not logfile_name:
    return logger

  if os.path.exists(work_dir):
    raise SystemExit(f'Work directory `{work_dir}` has already existed!\n'
                     f'Please specify another one.')
  os.makedirs(work_dir)

  # Save log message with all levels in log file.
  fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  return logger
