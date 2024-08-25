import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi import UploadFile
from unittest.mock import MagicMock
from io import BytesIO
from functions.knowledge import add_knowledge

# Mock variabel lingkungan
os.environ["EMBEDDING_MODEL"] = "test-model"
os.environ["PGVECTOR_CONNECTION_STRING"] = "test-connection-string"
os.environ["COLLECTION_NAME"] = "test-collection"

# Fungsi pembantu untuk membuat mock UploadFile
def create_mock_upload_file(filename, file_content):
    file = BytesIO(file_content.encode())
    return UploadFile(filename=filename, file=file)

@pytest.fixture
def mock_pdf_reader(mocker):
    # Mocking PdfReader
    mock_pdf = mocker.patch('functions.knowledge.PdfReader')
    mock_pdf_instance = mock_pdf.return_value
    mock_pdf_instance.pages = [MagicMock()]
    mock_pdf_instance.pages[0].extract_text.return_value = "This is a test PDF content"
    return mock_pdf

@pytest.fixture
def mock_text_splitter(mocker):
    # Mocking RecursiveCharacterTextSplitter
    mock_splitter = mocker.patch('functions.knowledge.RecursiveCharacterTextSplitter')
    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.create_documents.return_value = ["Document chunk"]
    return mock_splitter

@pytest.fixture
def mock_embeddings(mocker):
    # Mocking SentenceTransformerEmbeddings
    mock_embeddings = mocker.patch('functions.knowledge.SentenceTransformerEmbeddings')
    return mock_embeddings

@pytest.fixture
def mock_pgvector(mocker):
    # Mocking PGVector
    mock_pgvector = mocker.patch('functions.knowledge.PGVector')
    return mock_pgvector

def test_add_knowledge_with_pdf(mock_pdf_reader, mock_text_splitter, mock_embeddings, mock_pgvector):
    mock_file = create_mock_upload_file("test.pdf", "PDF Content")
    response = add_knowledge(mock_file)
    assert response == {"message": "knowledge has been added successfully"}

def test_add_knowledge_with_non_pdf():
    mock_file = create_mock_upload_file("test.txt", "Text Content")
    response = add_knowledge(mock_file)
    assert response == {"error": "Invalid file type. Please Upload a PDF file"}
