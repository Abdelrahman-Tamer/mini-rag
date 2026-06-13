from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseSignal
from controllers.NLPController import load_city_mappings
import re
import os

class DataController(BaseController):
    
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 # convert MB to bytes

    def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in self.app_settings.FILE_ALLOWED_EXTENSIONS:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):

        random_key = self.generate_random_string()
        project_path = ProjectController().get_project_path(project_id=project_id)

        cleaned_file_name = self.get_clean_file_name(
            orig_file_name=orig_file_name
        )

        new_file_path = os.path.join(
            project_path,
            random_key + "_" + cleaned_file_name
        )

        while os.path.exists(new_file_path):
            random_key = self.generate_random_string()
            new_file_path = os.path.join(
                project_path,
                random_key + "_" + cleaned_file_name
            )

        return new_file_path, random_key + "_" + cleaned_file_name

    def get_clean_file_name(self, orig_file_name: str):

        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())

        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name

    def detect_city_from_content(self, file_content: list) -> str:
        mappings = load_city_mappings()
        known_cities = mappings.get("known_cities", [])
        country_map = {k: v for k, v in mappings.items() if k != "known_cities" and isinstance(v, str)}

        full_text = " ".join([
            chunk.page_content for chunk in file_content if hasattr(chunk, "page_content")
        ]).lower()

        matched = [c for c in known_cities if c and c.lower() in full_text]
        if len(matched) == 1:
            return matched[0]

        matched_countries = [c for c in country_map if c in full_text]
        if len(matched_countries) == 1:
            return country_map[matched_countries[0]]

        return None

    def extract_city_type(self, filename: str) -> dict:
        name_without_ext = filename.rsplit(".", 1)[0]
        parts = name_without_ext.split("_")

        result = {"city": None, "doc_type": None}

        if len(parts) >= 2:
            result["city"] = parts[0].lower().strip()
            result["doc_type"] = "_".join(parts[1:]).lower().strip()

        return result


