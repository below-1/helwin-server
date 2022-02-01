from fono.config import MONGO_URI
from pymongo import MongoClient
from werkzeug.local import LocalProxy
from flask import g
from bson.objectid import ObjectId

client = MongoClient(MONGO_URI)

def get_db():
    if "mongo_db" not in g:
        g.mongo_db = client.fono
    return g.mongo_db

mongo_db = LocalProxy(get_db)
