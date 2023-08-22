from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from config.database import collection_name, collection_text
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from models.text import textClass
import librosa, io
import torch
import soundfile as sf
from datasets import load_dataset
import os
import logging


router = APIRouter()


@router.post("/audio")
async def upload_audio(audio_file: UploadFile = File(...)):
    # baca audio file
    content = await audio_file.read()

    # Simpan file audio ke MongoDB
    file_id = collection_name.insert_one({"content": content}).inserted_id

    # load model and processor
    processor = WhisperProcessor.from_pretrained(
        os.getenv("SMALL_WHISPER", default="openai/whisper-small")
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        os.getenv("SMALL_WHISPER", default="openai/whisper-small")
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="indonesian", task="transcribe"
    )
    model.config.forced_decoder_ids = forced_decoder_ids

    # Proses audio untuk pengenalan teks
    waveform, sampling_rate = librosa.load(io.BytesIO(content), sr=16000)
    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    if transcription:
        return {"file_id": str(file_id), "transcription": transcription[0]}

    else:
        raise HTTPException(status_code=500, detail="Speech recognition failed.")


@router.post("/text")
async def text_to_speech(request: textClass):
    # Load the pre-trained models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Get the text input from the user
    input_text = request.text

    # Validate input
    if not input_text.strip():
        raise HTTPException(
            status_code=400, detail="Invalid input: Text cannot be empty."
        )

    # Save input text to MongoDB
    collection_text.insert_one({"text": input_text})

    # Generate audio
    inputs = processor(text=input_text, return_tensors="pt")
    speech = model.generate_speech(
        inputs.input_ids, speaker_embeddings, vocoder=vocoder
    )

    # Save the audio in memory
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, speech.numpy(), samplerate=16000, format="wav")
    audio_buffer.seek(0)

    return Response(audio_buffer.getvalue(), media_type="audio/wav")
