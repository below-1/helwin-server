from environs import Env
import json
from bson.objectid import ObjectId

env = Env()
env.read_env()

APP_SECRET = env.str("APP_SECRET")
MONGO_URI = env.str("MONGO_URI")

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        else:
            return obj
