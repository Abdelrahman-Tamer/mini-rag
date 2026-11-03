import os
import sys

# Ensure repo_root, src, and api are in sys.path for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
src_dir = os.path.join(repo_root, "src")
api_dir = os.path.join(repo_root, "api")

for p in [repo_root, src_dir, api_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
