from pathlib import Path
from typing  import Union, Optional

class Utils:
    """
    Core utilities used across the project.
    """
    
    @classmethod
    def find_root(
        cls, 
        marker_file : Union[str, list[str]] = 'pyproject.toml', 
        start_path  : Optional[Path] = None
    ) -> Path:
        """Find project root by searching for a marker file.
        
        Args:
            marker_file : File(s) that indicate project root
            start_path  : Path to start search from (defaults to caller's location)
            
        Returns:
            Path to project root
            
        Raises:
            FileNotFoundError: If marker file not found in any parent directory
        """
        markers = [marker_file] if isinstance(marker_file, str) else marker_file
        path    = Path(start_path or Path(__file__)).resolve()
        
        for parent in [path, *path.parents]:
            if any((parent / marker).exists() for marker in markers):
                return parent
                
        raise FileNotFoundError(f"Could not find any of {markers}")