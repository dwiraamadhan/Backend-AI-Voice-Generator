from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["audio_ml"]
collection_audio = db["audio_file"]
collection_text = db["text_file"]
collection_questions = db["question"]
collection_knowledge = db["knowledge_file"]
