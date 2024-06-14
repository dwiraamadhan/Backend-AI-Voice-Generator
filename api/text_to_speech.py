from fastapi import APIRouter, HTTPException
from models.text import textClass
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import os, torch, io, base64
import soundfile as sf
from fastapi.responses import JSONResponse
from functions.generate_speech import generate_speech

router = APIRouter()


# endpoint for text to speech
@router.post("/text_to_speech")
async def text_to_speech(request: textClass):
    try:

        speech = generate_speech(request.text)
        return{
            "speech": speech
        }
        # # Load the pre-trained models
        # processor = SpeechT5Processor.from_pretrained(os.getenv("TEXT_TO_SPEECH"))
        # model = SpeechT5ForTextToSpeech.from_pretrained(os.getenv("TEXT_TO_SPEECH"))
        # vocoder = SpeechT5HifiGan.from_pretrained(os.getenv("TEXT_TO_SPEECH_HIFIGAN"))

        # # Load xvector containing speaker's voice characteristics from a dataset
        # embeddings_dataset = load_dataset(
        #     "Matthijs/cmu-arctic-xvectors", split="validation"
        # )
        # speaker_embeddings = torch.tensor(
        #     embeddings_dataset[7306]["xvector"]
        # ).unsqueeze(0)

        # # Get the text answer
        # input_text = request.text

        # # Validate input
        # if not input_text.strip():
        #     raise HTTPException(
        #         status_code=400, detail="Invalid input: Text cannot be empty."
        #     )

        # # Generate audio
        # inputs = processor(text=input_text, return_tensors="pt")
        # speech = model.generate_speech(
        #     inputs.input_ids, speaker_embeddings, vocoder=vocoder
        # )

        # # Save the audio in memory
        # audio_buffer = io.BytesIO()
        # sf.write(audio_buffer, speech.numpy(), samplerate=16000, format="wav")
        # audio_buffer.seek(0)

        # # Encode the audio data in base64
        # audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        # print(audio_base64)
        # return JSONResponse(content={"base64_audio": audio_base64})

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
