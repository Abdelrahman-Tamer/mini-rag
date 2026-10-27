import os
import sys

# Ensure src/ is in sys.path for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
src_dir = os.path.join(repo_root, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
