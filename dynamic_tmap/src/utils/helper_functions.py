import logging
from config import LOGGING_LEVEL, LOGGING_FORMAT, LOG_FILE_PATH

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOGGING_LEVEL),
        format=LOGGING_FORMAT,
        filename=LOG_FILE_PATH  # To log to a file
    )


def format_time(seconds):
      hours, rem = divmod(seconds, 3600)
      minutes, seconds = divmod(rem, 60)
      return f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"

def find_input_type(file_path):
      if file_path.endswith('csv'):
            return 'csv'
      elif file_path.endswith('txt'):
            return 'txt'
      elif file_path.endswith('cxsmiles'):
            return 'cxsmiles'
      else: 
            raise ValueError('Unsupported input file. Only .csv, .txt. and .cxsmiles files are supported')

def validate_data():
    pass

def i_o_operations():
    pass

import time
import os
import psutil
import threading
import logging
import time
import logging

# Create logger 
logger = logging.getLogger(__name__)
logger = logging.getLogger(__file__)

def find_input_type(file_path):
      if file_path.endswith('csv'):
        return 'csv'
      elif file_path.endswith('txt'):
        return 'txt'
      elif file_path.endswith('cxsmiles'):
        return 'cxsmiles'
      elif file_path.endswith('parquet'):
        return 'parquet'
      else: 
        raise ValueError('Unsupported input file. Only .csv, .txt. .parquet and .cxsmiles files are supported')

def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} h; {int(minutes)}m; {seconds:.2f}s"

def process_input(input_path):
    if os.path.isdir(input_path):
        # Process all files in the directory
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            if os.path.isfile(file_path):  # Optional: filter by file extensions
                yield file_path
    elif os.path.isfile(input_path):
        # Process a single file
        yield input_path
    else:
        raise ValueError(f"Invalid input path: {input_path}")

class TimeTracker:
    def __init__(self, description="", logger=None):
        self.description = description
        self.logger = logger or logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.description} started.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.logger.info(f"{self.description} completed.")
        self.logger.info(f"Total time elapsed: {format_time(self.elapsed_time)}")

class RAMTracker:
    def __init__(self, description="", interval=1.0, logger=None):
        self.description = description
        self.interval = interval
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self.max_ram = 0
        self.logger = logger or logging.getLogger(__name__)

    def _monitor_ram(self):
        while not self._stop_event.is_set():
            ram_usage = self.process.memory_info().rss / (1024 ** 2)  # Convert to MB
            if ram_usage > self.max_ram:
                self.max_ram = ram_usage
            time.sleep(self.interval)

    def __enter__(self):
        self.logger.info(f"{self.description} RAM monitoring started.")
        self._thread = threading.Thread(target=self._monitor_ram)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        self.logger.info(f"{self.description} RAM monitoring completed.")
        self.logger.info(f"Peak RAM usage: {self.max_ram:.2f} MB")

class RAMAndTimeTracker:
    def __init__(self, description="", interval=1.0, logger=None, display_progress=True):
        self.description = description
        self.interval = interval
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self.max_ram = 0
        self.logger = logger or logging.getLogger(__name__)
        self.display_progress = display_progress

    def _monitor(self):
        while not self._stop_event.is_set():
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            ram_usage = self.process.memory_info().rss / (1024 ** 2)  # Convert to MB
            if ram_usage > self.max_ram:
                self.max_ram = ram_usage
            if self.display_progress:
                print(
                    f"\r{self.description}: Elapsed time: {elapsed_time:.2f}s, "
                    f"Current RAM: {ram_usage:.2f} MB, Peak RAM: {self.max_ram:.2f} MB",
                    end=""
                )
            time.sleep(self.interval)

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.description} started.")
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.info(f"{self.description} completed.")
        self.logger.info(f"Total time elapsed: {format_time(elapsed_time)}")
        self.logger.info(f"Peak RAM usage: {self.max_ram:.2f} MB")
        if self.display_progress:
            print()  # Move to the next line after the progress output

class ProgressTracker:
    def __init__(self, description="", total_steps=1, interval=1.0, logger=None, display_progress=True):
        self.description = description
        self.total_steps= total_steps
        self.interval = interval
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self.max_ram = 0
        self.current_file = 0
        self.logger = logger or logging.getLogger(__name__)
        self.display_progress = display_progress
        self.start_time = None
        self._thread = None

    def _update_loading_bar(self, current):
        bar_length = 30
        filled_length = int(round(bar_length * current / float(self.total_steps)))
        percents = round(100.0 * current / float(self.total_steps), 1)
        bar = '#' * filled_length + '>' + '-' * (bar_length - filled_length)
        return f"{self.description}: |{bar}| {current}/{self.total_steps} ({percents}%)"

    def _monitor(self):
        while not self._stop_event.is_set():
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            ram_usage = self.process.memory_info().rss / (1024 ** 3) # Convert to GB
            total_mem = psutil.virtual_memory()
            system_ram = f"Mem: {total_mem.used / (1024 ** 3):.1f}G/{total_mem.total / (1024 ** 3):.1f}G"
            if total_mem.used> self.max_ram:
                # self.max_ram = ram_usage
                self.max_ram = total_mem.used
            if self.display_progress:
                loading_bar = self._update_loading_bar(self.current_file)
                print(
                f"\r{loading_bar} Time: {format_time(elapsed_time)}, "
                f"Peak RAM: {self.max_ram / (1024 ** 3):.1f} GB, {system_ram}",
                end=""
                )
            time.sleep(self.interval)

    def update_progress(self):
        self.current_file += 1

    def __enter__(self):
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.info(f"{self.description} completed.")
        self.logger.info(f"Total time elapsed: {format_time(elapsed_time)}")
        self.logger.info(f"Peak RAM usage: {self.max_ram:.2f} MB")
        if self.display_progress:
            print()  # Move to the next line after the progress output

