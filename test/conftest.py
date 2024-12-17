import pytest
from pathlib import Path
import sys

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add any pytest fixtures here if needed
