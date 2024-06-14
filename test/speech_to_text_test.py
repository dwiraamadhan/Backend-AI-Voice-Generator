import pytest
import os
from unittest import mock
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from functions.transcribe import transcribe_audio_file


@pytest.fixture
def mock_processor():
    processor = mock.MagicMock(spec=WhisperProcessor)
    processor.get_decoder_prompt_ids.return_value = [(0, 50257), (1, 50257)]
    processor.batch_decode.return_value = ["transcribed text"]
    return processor


@pytest.fixture
def mock_model():
    model = mock.MagicMock(spec=WhisperForConditionalGeneration)
    model.generate.return_value = [[50257]]

    # add attribute config
    config_mock = mock.MagicMock()
    model.config = config_mock
    return model


@pytest.fixture
def mock_librosa():
    with mock.patch("librosa.load") as librosa_load:
        librosa_load.return_value = (mock.MagicMock(), 16000)
        yield librosa_load


@pytest.fixture
def mock_collection():
    with mock.patch("config.database.collection_audio.insert_one") as mock_insert:
        mock_insert.return_value.inserted_id = "mock_file_id"
        yield mock_insert


@pytest.mark.asyncio
async def test_transcribe_audio_file(
    mock_processor, mock_model, mock_librosa, mock_collection
):
    with mock.patch("transformers.WhisperProcessor.from_pretrained") as mock_proc_from_pretrained, mock.patch("transformers.WhisperForConditionalGeneration.from_pretrained") as mock_model_from_pretrained:
        mock_proc_from_pretrained.return_value = mock_processor
        mock_model_from_pretrained.return_value = mock_model

        # Mock audio content
        content = b"fake audio content"

        # Call the function
        file_id, transcription = await transcribe_audio_file(content)

        # Aserstion
        assert file_id == "mock_file_id"
        assert transcription == ["transcribed text"]
        mock_proc_from_pretrained.assert_called_once_with(os.getenv("SMALL_WHISPER"))
        mock_model_from_pretrained.assert_called_once_with(os.getenv("SMALL_WHISPER"))
        mock_processor.get_decoder_prompt_ids.assert_called_once_with(language="english", task="transcribe")
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()