from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores.pgvector import PGVector
# from langchain_community.vectorstores.pinecone import Pinecone

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
        temperature=0.2,
        do_Sample=True,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

    # create custom prompt
    custom_prompt_template = """
        Please use the following pieces of information
        to answer the user's question. If you dont know the answer, please just say that you don't
        know the answer, don't try to make up an answer.

        Context: {context}
        Question:{question}  
"""

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    

    db = PGVector(
        collection_name=os.getenv("COLLECTION_NAME"),
        connection_string=os.getenv("PGVECTOR_CONNECTION_STRING"),
        embedding_function=embeddings,
    )

    # os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    # db = Pinecone.from_existing_index(
    #     index_name = os.getenv("PINECONE_INDEX_NAME"),
    #     embedding = embeddings
    # )

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa


def process_answer(instruction):
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text["result"]
    return answer
