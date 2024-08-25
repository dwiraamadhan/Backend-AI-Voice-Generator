import os
# from fastapi import UploadFile
# from docx import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.pgvector import PGVector
from fastapi import UploadFile


def add_knowledge(knowledge_file: UploadFile):
    # read the document
    # document = Document(knowledge_file.file)
    # knowledge_text = ""
    # for paragraph in document.paragraphs:
    #     knowledge_text += paragraph.text + " "
    #     print(knowledge_text)

    # check file extenstion
    if knowledge_file.filename.endswith(".pdf"):
        knowledge_text = ""
        reader = PdfReader(knowledge_file.file)
        for page in reader.pages:
            knowledge_text += page.extract_text()

        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        print("text splitter created")

        # convert the chunks into documents
        docs = text_splitter.create_documents([knowledge_text])
        print("text pecahan sudah dibuat menjadi dokumen")
        print(len(docs))

        # load Embeddings model
        embeddings = SentenceTransformerEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL")
        )

        # create db instance to save knowledge
        # CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=os.getenv("COLLECTION_NAME"),
            connection_string=os.getenv("PGVECTOR_CONNECTION_STRING"),
        )
        print("vector store PGVector sudah di load")

        return {
            "message": "knowledge has been added successfully"
        }
    
    else:
        return {
            "error": "Invalid file type. Please Upload a PDF file"
        }