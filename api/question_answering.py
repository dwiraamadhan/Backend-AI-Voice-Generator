from fastapi import APIRouter, HTTPException
from models.question import QuestionClass
from functions.question_answering import process_answer
from datetime import datetime
from config.database import collection_questions

router = APIRouter()


@router.post("/question")
async def answering(question: QuestionClass):
    # Process question
    result = process_answer(question.text)

    # Save question to database
    question_doc = {"text": question.text, "createdAt": datetime.now()}
    question_text = collection_questions.insert_one(question_doc)
    inserted_id = str(question_text.inserted_id)

    return {
        "relevant_answer": result,
        "questions_saved": {
            "question_id": inserted_id,
            "question_text": question.text,
            "createdAt": question.createdAt,
        },
    }
