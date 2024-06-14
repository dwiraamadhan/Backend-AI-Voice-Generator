import pytest
from unittest.mock import patch, MagicMock
import torch, os, sys, base64
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.generate_speech import generate_speech

# Sample data for mocking
sample_text = "Hello, this is a test."
sample_speaker_embedding = torch.rand((1, 512))  # Assuming the xvector dimension is 512
sample_speech_output = torch.rand((16000,))  # Assuming 1 second of audio at 16kHz

# Mocking soundfile.write to write to a buffer
def mock_sf_write(file, data, samplerate, format):
    file.write(b'RIFF....WAVEfmt...')  # Simplified wave header for testing

@pytest.fixture
def mock_environment():
    env_vars = {
        "TEXT_TO_SPEECH": "mock_model",
        "TEXT_TO_SPEECH_HIFIGAN": "mock_vocoder",
    }
    with patch.dict(os.environ, env_vars):
        yield

@patch('functions.generate_speech.SpeechT5Processor.from_pretrained')
@patch('functions.generate_speech.SpeechT5ForTextToSpeech.from_pretrained')
@patch('functions.generate_speech.SpeechT5HifiGan.from_pretrained')
@patch('functions.generate_speech.load_dataset')
@patch('functions.generate_speech.sf.write', side_effect=mock_sf_write)
def test_generate_speech(mock_load_dataset, mock_hifi_gan, mock_text_to_speech, mock_processor, mock_environment):
    # Mock the processor and models
    mock_processor.return_value = MagicMock()
    mock_text_to_speech.return_value = MagicMock(generate_speech=MagicMock(return_value=sample_speech_output))
    mock_hifi_gan.return_value = MagicMock()
    
    # Mock the dataset
    mock_load_dataset.return_value = MagicMock()
    mock_load_dataset.return_value[7306] = {"xvector": sample_speaker_embedding.squeeze(0).numpy()}

    # Valid input test
    result = generate_speech(sample_text)
    decoded_audio = base64.b64decode(result)
    assert decoded_audio.startswith(b'RIFF'), "The result should be a WAV file encoded in base64."