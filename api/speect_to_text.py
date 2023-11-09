from fastapi import APIRouter, UploadFile, File, HTTPException
from config.database import collection_audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, io, os

router = APIRouter()


# endpoint for transcribing audio to text
@router.post("/speech_to_text")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        # read audio file
        content = await audio_file.read()

        # save audio file to MongoDB
        file_id = collection_audio.insert_one({"content": content}).inserted_id

        # load model and processor
        processor = WhisperProcessor.from_pretrained(os.getenv("SMALL_WHISPER"))
        model = WhisperForConditionalGeneration.from_pretrained(
            os.getenv("SMALL_WHISPER")
        )
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        model.config.forced_decoder_ids = forced_decoder_ids

        # transcribing process
        waveform, sampling_rate = librosa.load(io.BytesIO(content), sr=16000)
        input_features = processor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return {"file_id": str(file_id), "transcription": transcription[0]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
