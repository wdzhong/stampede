import os
from datetime import datetime


class Logger(object):
    def __init__(self, log_dir, timestamp=None):
        """Create a summary writer logging to log_dir."""
        if not timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.timestamp = timestamp
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = '{:}/log-{:}.txt'.format(self.log_dir, timestamp)
        self.file_writer = open(self.log_file, 'w')

    def print(self, string, fprint=True):
        """
        print the string to terminal and save it to file if specified.
        """
        print(string)
        if fprint:
            self.file_writer.write(f'{string}\n')
            self.file_writer.flush()

    def close(self):
        self.file_writer.close()
