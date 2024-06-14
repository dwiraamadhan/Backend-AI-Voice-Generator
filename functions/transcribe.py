from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, io, os
from config.database import collection_audio

async def transcribe_audio_file(content: bytes) -> str:
    try:
        # Save audio file to MongoDB and get file ID
        file_id = collection_audio.insert_one({"content": content}).inserted_id

        # Load model and processor
        processor = WhisperProcessor.from_pretrained(os.getenv("SMALL_WHISPER"))
        model = WhisperForConditionalGeneration.from_pretrained(
            os.getenv("SMALL_WHISPER")
        )
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        model.config.forced_decoder_ids = forced_decoder_ids

        # Transcribing process
        waveform, sampling_rate = librosa.load(io.BytesIO(content), sr=16000)
        input_features = processor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return str(file_id), transcription

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")
