from unittest.mock import MagicMock
import os
import pytest
from controllers.DataController import DataController
from models import ResponseSignal


@pytest.fixture
def data_controller():
    return DataController()


def test_clean_file_name_path_traversal(data_controller):
    # Test UNIX directory traversal
    name1 = data_controller.get_clean_file_name("../../etc/passwd")
    assert ".." not in name1
    assert "/" not in name1
    assert name1 == "passwd"

    # Test Windows directory traversal
    name2 = data_controller.get_clean_file_name("..\\..\\Windows\\System32\\cmd.exe")
    assert ".." not in name2
    assert "\\" not in name2
    assert name2 == "cmd.exe"


def test_clean_file_name_special_characters(data_controller):
    cleaned = data_controller.get_clean_file_name("tour #1 (Cairo & Giza)!.pdf")
    assert "#" not in cleaned
    assert "(" not in cleaned
    assert ")" not in cleaned
    assert "!" not in cleaned
    assert cleaned.endswith(".pdf")


def test_clean_file_name_empty(data_controller):
    assert data_controller.get_clean_file_name("") == "file"
    assert data_controller.get_clean_file_name(None) == "file"


def test_validate_uploaded_file_valid(data_controller):
    mock_file = MagicMock()
    mock_file.content_type = "application/pdf"
    mock_file.filename = "egypt_tourist_guide.pdf"
    mock_file.size = 1024 * 1024  # 1 MB

    is_valid, signal = data_controller.validate_uploaded_file(mock_file)
    assert is_valid is True
    assert signal == ResponseSignal.FILE_VALIDATED_SUCCESS.value


def test_validate_uploaded_file_invalid_type(data_controller):
    mock_file = MagicMock()
    mock_file.content_type = "application/x-executable"
    mock_file.filename = "malicious.exe"
    mock_file.size = 1024

    is_valid, signal = data_controller.validate_uploaded_file(mock_file)
    assert is_valid is False
    assert signal == ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value


def test_validate_uploaded_file_size_exceeded(data_controller):
    mock_file = MagicMock()
    mock_file.content_type = "application/pdf"
    mock_file.filename = "huge_manual.pdf"
    mock_file.size = 50 * 1024 * 1024  # 50 MB > 10 MB limit

    is_valid, signal = data_controller.validate_uploaded_file(mock_file)
    assert is_valid is False
    assert signal == ResponseSignal.FILE_SIZE_EXCEEDED.value


def test_unique_filepath_generation(data_controller):
    full_path, file_id = data_controller.generate_unique_filepath("cairo_guide.txt")
    assert file_id.endswith("cairo_guide.txt")
    assert os.path.isabs(full_path)
    assert full_path.endswith(file_id)
