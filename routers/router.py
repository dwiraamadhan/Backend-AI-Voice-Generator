from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from config.database import (
    collection_name,
    collection_knowledge,
    collection_questions,
)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    pipeline,
)
from models.text import textClass
from models.question import QuestionClass
import librosa, io
import torch
import soundfile as sf
from datasets import load_dataset
import os
from docx import Document
from datetime import datetime
from functions.questions_answer import (
    extract_bracketed_sentences,
    separate_brackets,
)
import base64

# from fastapi.responses import StreamingResponse
# from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
# from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


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

    # Generate audio
    inputs = processor(text=input_text, return_tensors="pt")
    speech = model.generate_speech(
        inputs.input_ids, speaker_embeddings, vocoder=vocoder
    )

    # Save the audio in memory
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, speech.numpy(), samplerate=16000, format="wav")
    audio_buffer.seek(0)

    # Encode the audio data in base64
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    print(audio_base64)
    return JSONResponse(content={"base64_audio": audio_base64})


@router.post("/knowledge")
async def update_knowledge(knowledge_file: UploadFile = File(...)):
    try:
        document = Document(knowledge_file.file)
        knowledge_text = ""
        for paragraph in document.paragraphs:
            knowledge_text += paragraph.text + " "

        knowledge_text = separate_brackets(knowledge_text)

        # check if knowledge already exists
        existing_knowledge = collection_knowledge.find_one()
        if existing_knowledge:
            # if existing knowledge exists, remove it based on ID
            collection_knowledge.delete_one({"_id": existing_knowledge["_id"]})
            print("knowledge removed")

        # add new knowledge
        collection_knowledge.insert_one({"text": knowledge_text})
        print("knowledge added")
        return {
            "message": "Knowledge context added successfully",
            "knowledge_text": knowledge_text,
        }

    except Exception as e:
        raise HTTPException(detail=e.args[0], status_code=400)


@router.post("/question")
async def question_answer(question: QuestionClass):
    try:
        # Setup Question Answering model
        qa_pipeline = pipeline(
            "question-answering", model="timpal0l/mdeberta-v3-base-squad2"
        )

        context = ""
        for doc in collection_knowledge.find({}, {"text": 1}):
            context += doc["text"] + " "
            print(context)

        result = qa_pipeline({"context": context, "question": question.text})
        answer = result["answer"]

        target_sentence = answer
        target_index = result["start"]

        matching_sentences = extract_bracketed_sentences(
            context, target_sentence, target_index
        )

        if len(matching_sentences) > 0:
            print("Matching bracketed sentences:")
            for sentence in matching_sentences:
                print(sentence.strip())

        else:
            print("No matching bracketed sentences found.")

        # Save question to database
        question_doc = {
            "text": question.text,
            "createdAt": datetime.now(),
        }

        question_text = collection_questions.insert_one(question_doc)
        inserted_id = str(question_text.inserted_id)

        return {
            "relevant_answer": matching_sentences,
            "questions_saved": {
                "question_id": inserted_id,
                "question_text": question.text,
                "createdAt": question.createdAt,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])


# @router.post("/tts")
# async def text_to_speech(text: str):
#     models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
#         "facebook/fastspeech2-en-ljspeech",
#         arg_overrides={"vocoder": "hifigan", "fp16": False},
#     )

#     TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
#     generator = task.build_generator(models, cfg)

#     sample = TTSHubInterface.get_model_input(task, text)
#     wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)

#     # Convert audio data to bytes
#     audio_bytes = io.BytesIO(wav.tobytes())

#     return StreamingResponse(io.BytesIO(audio_bytes.read()), media_type="audio/wav")
