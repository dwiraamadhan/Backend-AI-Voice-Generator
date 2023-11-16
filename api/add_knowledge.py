from fastapi import APIRouter, UploadFile, File, HTTPException
from docx import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import SentenceTransformerEmbeddings
import os
from langchain.vectorstores.pgvector import PGVector


router = APIRouter()


@router.post("/knowledge")
async def add_knowledge(
    knowledge_file: UploadFile = File(...),
):
    try:
        # read the document
        document = Document(knowledge_file.file)
        knowledge_text = ""
        for paragraph in document.paragraphs:
            knowledge_text += paragraph.text + " "
        print(knowledge_text)

        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        print("file sudah di pecah menjadi beberapa text")

        # convert the chunks into documents
        docs = text_splitter.create_documents([knowledge_text])
        print("text pecahan sudah dibuat menjadi dokumen")

        # print(texts)
        print(len(docs))

        # load Embeddings model
        embeddings = SentenceTransformerEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL")
        )
        print("embeddings sudah di load")

        db = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=os.getenv("COLLECTION_NAME"),
            connection_string=os.getenv("CONNECTION_STRING"),
        )

        print("vector store PGVector sudah di load")

        return {
            "message": "knowledge has been added successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
