import logging

class Logger():

    LOG_FILE = "log.txt"

    def setupLogger(self, name, logFile):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create a file handler
        handler = logging.FileHandler(logFile)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
        handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(handler)

        return logger
    
    def getLogger(self, name, logFile=None):
        # Set log file path
        if logFile:
            self.LOG_FILE = logFile
        return self.setupLogger(name, self.LOG_FILE)     