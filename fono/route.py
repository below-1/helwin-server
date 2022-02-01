import json
from flask import (
    Blueprint,
    request,
    current_app
)
import pymongo
from fono.mongo import mongo_db
from fono.config import Encoder


bp = Blueprint("api", __name__, url_prefix="/api/v1")


@bp.route('kata-tokens/<kata>', methods=['GET'])
def find_kata_tokens(kata : str):
    result = mongo_db.data.find_one({
        'word': kata
    })
    resp = current_app.response_class(
        response=json.dumps(result, cls=Encoder),
        status=200,
        mimetype='application/json'
    )
    return resp

@bp.route('kata/<kata>', methods=['GET'])
def find_kata(kata : str):
    query_obj = {
        'word': {
            '$regex': '^' + kata + '.*'
        }
    }
    projection_obj = {
        'word': 1
    }
    result = list( mongo_db.data.find(query_obj, projection_obj).sort('word', pymongo.ASCENDING).limit(10) )
    resp = current_app.response_class(
        response=json.dumps(result, cls=Encoder),
        status=200,
        mimetype='application/json'
    )
    return resp


@bp.route('letters/<tag>', methods=['GET'])
def chars(tag: str):
    query = {
        'tag': tag
    }
    result = list( mongo_db.counter.find(query).sort('letter', pymongo.ASCENDING) )
    resp = current_app.response_class(
        response=json.dumps(result, cls=Encoder),
        status=200,
        mimetype='application/json'
    )
    return resp

@bp.route("fiturs/<tag>", methods=['GET'])
def fiturs(tag: str):
    query = {
        'tag': tag
    }
    result = list( mongo_db.fiturs.find(query).sort('idx', pymongo.ASCENDING) )
    resp = current_app.response_class(
        response=json.dumps(result, cls=Encoder),
        status=200,
        mimetype='application/json'
    )
    return resp
