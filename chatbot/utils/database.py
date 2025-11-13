# Mongo DB for logging
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
# retrieve mongo uri (connection string) from .env
MONGO_URI = os.getenv("MONGO_URI")
# creates a connection object to MongoDB
mongo = MongoClient(MONGO_URI)
# creates/opens a database named voice_db
db = mongo["voice_db"]

def get_people_collection():
  return db["people"]


