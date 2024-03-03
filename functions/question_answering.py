from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
# from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.pinecone import Pinecone
import pinecone

# load model question answering
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, device_map="auto", offload_folder="offload", torch_dtype=torch.float32
)


def llm_pipeline():
    # initialize pipeline
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.3,
        do_Sample=True,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    # CONNECTION_STRING = PGVector.connection_string_from_db_params(
    #         driver = os.getenv("PGVECTOR_DRIVER"),
    #         host = os.getenv("PGVECTOR_HOST"),
    #         port = os.getenv("PGVECTOR_PORT"),
    #         database = os.getenv("PGVECTOR_DATABASE"),
    #         user = os.getenv("PGVECTOR_USER"),
    #         password = os.getenv("PGVECTOR_PASSWORD"),
    #     )
    # db = PGVector(
    #     collection_name=os.getenv("COLLECTION_NAME"),
    #     connection_string=os.getenv("CONNECTION_STRING"),
    #     embedding_function=embeddings,
    # )

    # pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
    db = Pinecone.from_existing_index(
        index_name = os.getenv("PINECONE_INDEX_NAME"),
        embedding = embeddings
    )

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    return qa


def process_answer(instruction):
    response = ""
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text["result"]
    return answer
