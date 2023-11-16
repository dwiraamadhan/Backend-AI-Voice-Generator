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
from langchain.vectorstores import Milvus, ElasticsearchStore
import os
from langchain.vectorstores.pgvector import PGVector


router = APIRouter()


@router.post("/knowledge")
async def add_knowledge(
    knowledge_file: UploadFile = File(...),
):
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
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    print("embeddings sudah di load")

    # # create connection to milvus to check existed collection
    # milvus_connect = connections.connect(
    #     host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT")
    # )
    # if utility.has_collection(os.getenv("MILVUS_COLLECTION")):
    #     utility.drop_collection(os.getenv("MILVUS_COLLECTION"))

    # # save vector data to database Milvus
    # vector_store = Milvus.from_documents(
    #     docs,
    #     embedding=embeddings,
    #     collection_name=os.getenv("MILVUS_COLLECTION"),
    #     connection_args={
    #         "host": os.getenv("MILVUS_HOST"),
    #         "port": os.getenv("MILVUS_PORT"),
    #     },
    # )

    # vector_store = ElasticsearchStore(
    #     embedding=embeddings,
    #     index_name="chatbotBNIDirect",
    #     es_user="magang",
    #     es_password="magang12345",
    #     es_url="http://154.41.251.22:9200",
    # )

    # CONNECTION_STRING = PGVector.connection_string_from_db_params(
    #     driver=os.getenv("PGVECTOR_DRIVER"),
    #     host=os.getenv("PGVECTOR_HOST"),
    #     port=os.getenv("PGVECTOR_PORT"),
    #     database=os.getenv("PGVECTOR_DATABASE"),
    #     user=os.getenv("PGVECTOR_USER"),
    #     password=os.getenv("PGVECTOR_PASSWORD"),
    # )

    COLLECTION_NAME = "chatbotBNIDirect"
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=os.getenv("CONNECTION_STRING"),
    )

    print("vector store milvus sudah di load")

    return {
        "message": "knowledge has been added successfully",
    }
