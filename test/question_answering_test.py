import pytest
from unittest.mock import patch, MagicMock
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.question_answering import process_answer

@pytest.fixture
def mock_environment():
    env_vars = {
        "EMBEDDING_MODEL": "mock_model",
        "PGVECTOR_CONNECTION_STRING": "mock_connection_string",
        "COLLECTION_NAME": "mock_collection"
    }
    with patch.dict(os.environ, env_vars):
        yield

@patch("functions.question_answering.pipeline")
@patch("functions.question_answering.HuggingFacePipeline")
@patch("functions.question_answering.SentenceTransformerEmbeddings")
@patch("functions.question_answering.PGVector")
def test_process_answer(
    mock_pgvector,
    mock_sentence_embeddings,
    mock_huggingface_pipeline,
    mock_pipeline,
    mock_environment
):
    # Mock pipeline
    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance

    # Mock SentenceTransformerEmbeddings
    mock_sentence_embeddings_instance = MagicMock()
    mock_sentence_embeddings.return_value = mock_sentence_embeddings_instance

    # Mock PGVector
    mock_pgvector_instance = MagicMock()
    mock_pgvector.return_value = mock_pgvector_instance

    # Mock retriever and QA chain
    mock_retriever = MagicMock()
    mock_pgvector_instance.as_retriever.return_value = mock_retriever    
    mock_qa_instance = MagicMock()
    mock_qa_instance.return_value = {"result": "mock_answer"}

    with patch("functions.question_answering.RetrievalQA.from_chain_type", return_value=mock_qa_instance):
        # Call the function
        instruction = "What is the interest rate for savings accounts?"
        answer = process_answer(instruction)

        # Assertions
        assert answer == "mock_answer"
        mock_huggingface_pipeline.assert_called_once_with(pipeline=mock_pipeline_instance)
        mock_sentence_embeddings.assert_called_once_with(model_name="mock_model")
        mock_pgvector.assert_called_once_with(
            collection_name="mock_collection",
            connection_string="mock_connection_string",
            embedding_function=mock_sentence_embeddings_instance
        )
        mock_pgvector_instance.as_retriever.assert_called_once()
        mock_qa_instance.assert_called_once_with(instruction)