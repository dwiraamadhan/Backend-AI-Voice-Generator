from fastapi import APIRouter, UploadFile, File, HTTPException
from pymilvus import (
    utility,
    connections,
)
from docx import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Milvus
import os


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

        # create connection to milvus to check existed collection
        milvus_connect = connections.connect(
            host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT")
        )
        if utility.has_collection(os.getenv("MILVUS_COLLECTION")):
            utility.drop_collection(os.getenv("MILVUS_COLLECTION"))

        # save vector data to database Milvus
        vector_store = Milvus.from_documents(
            docs,
            embedding=embeddings,
            collection_name=os.getenv("MILVUS_COLLECTION"),
            connection_args={
                "host": os.getenv("MILVUS_HOST"),
                "port": os.getenv("MILVUS_PORT"),
            },
        )
        print("vector store milvus sudah di load")

        return {
            "message": "knowledge has been added successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=e.args[0])
