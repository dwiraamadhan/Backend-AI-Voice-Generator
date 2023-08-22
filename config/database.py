from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["audio_ml"]
collection_name = db["audio_file"]
collection_text = db["text_file"]
