from helpers.config import get_settings, Settings
import os
import random
import string
import tempfile

class BaseController:
    
    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        
        is_serverless = (
            os.environ.get("VERCEL") == "1"
            or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
            or not os.access(self.base_dir, os.W_OK)
        )
        
        if is_serverless:
            self.files_dir = os.path.join(tempfile.gettempdir(), "nileguide_assets_files")
        else:
            self.files_dir = os.path.join(self.base_dir, "assets/files")
            
        try:
            os.makedirs(self.files_dir, exist_ok=True)
        except Exception:
            self.files_dir = os.path.join(tempfile.gettempdir(), "nileguide_assets_files")
            os.makedirs(self.files_dir, exist_ok=True)

    def generate_random_string(self, length: int = 12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
