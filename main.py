from fastapi import FastAPI
from api.update_knowledge import router as router_knowledge
from api.question_answering import router as router_qa
from api.speech_to_text import router as router_s2t
# from api.text_to_speech import router as router_t2s
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

load_dotenv()
app = FastAPI()

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a new client and connect to the server
uri = "mongodb://localhost:27017/"
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

app.include_router(router_knowledge)
app.include_router(router_qa)
app.include_router(router_s2t)
# app.include_router(router_t2s)
