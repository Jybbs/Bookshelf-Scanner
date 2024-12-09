import logging

from bookshelf_scanner.core.utils import Utils

class ModuleLogger:
    """
    Configures and manages logging for Bookshelf Scanner modules.
    """
    PROJECT_ROOT = Utils.find_root('pyproject.toml')
    LOGS_DIR     = PROJECT_ROOT / 'bookshelf_scanner' / 'logs'

    def __init__(self, module_name: str):
        """
        Initialize logger configuration for a specific module.
        
        Args:
            module_name : Name of the module requesting the logger
        """
        self.logger   = logging.getLogger(f'bookshelf_scanner.{module_name}')
        self.log_file = self.LOGS_DIR / f'{module_name}.log'
        
        self.configure_logger()
    
    def configure_logger(self):
        """
        Sets up logger with file handler if not already configured.
        """
        if not self.logger.handlers:

            self.logger.setLevel(logging.INFO)
            self.LOGS_DIR.mkdir(exist_ok = True)
            
            handler = logging.FileHandler(self.log_file, mode = 'a')
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def __call__(self) -> logging.Logger:
        """
        Returns the configured logger instance.
        """
        return self.logger