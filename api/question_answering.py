from fastapi import APIRouter, HTTPException
from models.question import QuestionClass
from functions.question_answering import process_answer
from functions.generate_speech import generate_speech
from datetime import datetime
from config.database import collection_questions

router = APIRouter()


@router.post("/question")
async def answering(question: QuestionClass):
    try:
        # Process question
        result = process_answer(question.text)
        print("Answer: ", result)

        # Save question to database
        question_doc = {"text": question.text, "createdAt": datetime.now()}
        question_text = collection_questions.insert_one(question_doc)
        inserted_id = str(question_text.inserted_id)

        # generate speech
        base64_audio = generate_speech(result)
        print("base64_audio: ", base64_audio)

        return {
            "relevant_answer": result,
            "base64_audio": base64_audio,
            "questions_saved": {
                "question_id": inserted_id,
                "question_text": question.text,
                "createdAt": question.createdAt,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
