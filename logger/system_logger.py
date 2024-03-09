import logging
class SystemLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def log_info(self, message):
        logging.info(message)

    def log_warning(self, message):
        logging.warning(message)

    def log_error(self, message):
        logging.error(message)

    def log_critical(self, message):
        logging.critical(message)