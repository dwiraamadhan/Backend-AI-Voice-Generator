from fastapi import APIRouter, UploadFile, File, HTTPException
from docx import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from langchain.vectorstores.pgvector import PGVector
from functions.knowledge import add_knowledge
# from langchain_community.vectorstores.pinecone import Pinecone
# from langchain_pinecone import Pinecone


router = APIRouter()


@router.post("/knowledge")
async def update_knowledge(
    document: UploadFile = File(...),
):
    try:
        result = add_knowledge(document)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
