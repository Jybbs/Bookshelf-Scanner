# ModuleLogger

The `ModuleLogger` module provides consistent logging configuration across the Bookshelf Scanner project. It ensures all modules use standardized log formatting, file handling, and directory organization.

## Core Features

- **Standardized Configuration**
  - Consistent log formatting across all project modules
  - Automatic log directory creation and management
  - Module-specific log files for easy debugging

- **Project-Aware Paths**
  - Automatic project root detection
  - Centralized logs directory within project structure
  - Path handling compatible with different environments

## Usage

```python
from bookshelf_scanner import ModuleLogger

# Initialize logger for your module
logger = ModuleLogger(__name__)()

# Use standard logging levels
logger.debug("Detailed information for debugging")
logger.info("General information about program execution")
logger.warning("Warning messages for potentially problematic situations")
logger.error("Error messages for serious problems")
```

### Log File Structure

Each module gets its own log file in the `bookshelf_scanner/logs` directory:
```
bookshelf_scanner/
├── logs/
│   ├── extractor.log
│   ├── segmenter.log
│   └── optimizer.log
```

### Log Format

Each log entry follows a consistent format:
```
2024-01-15 14:30:45,123 - INFO - Processing started
2024-01-15 14:30:45,234 - ERROR - Failed to process image: file not found
```