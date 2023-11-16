from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Milvus, ElasticsearchStore
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
from elasticsearch import Elasticsearch
from langchain.vectorstores.pgvector import PGVector

# load model question answering
checkpoint = "LaMini-Flan-T5-783M"
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

    # initialize vector store Milvus
    # db = Milvus(
    #     embeddings,
    #     collection_name=os.getenv("MILVUS_COLLECTION"),
    #     connection_args={
    #         "host": os.getenv("MILVUS_HOST"),
    #         "port": os.getenv("MILVUS_PORT"),
    #     },
    # )

    # es_connection = Elasticsearch(
    #     "http://154.41.251.22:9200",
    #     bearer_auth="UWt5cnpJc0JLNndiZXBhYlhHOTk6Ul9jVEJzenlUQy1WVURfUy1PMHBEQQ==",
    #     # basic_auth=("magang", "magang12345"),
    # )

    # db = ElasticsearchStore(
    #     embedding=embeddings,
    #     index_name="chatbotBNIDirect",
    #     es_connection=es_connection,
    # )

    COLLECTION_NAME = "chatbotBNIDirect"
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=os.getenv("CONNECTION_STRING"),
        embedding_function=embeddings,
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
