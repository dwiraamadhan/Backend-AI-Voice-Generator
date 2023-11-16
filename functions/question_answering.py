from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
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
    db = PGVector(
        collection_name=os.getenv("COLLECTION_NAME"),
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
