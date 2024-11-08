from flask import request, Response
from bson import json_util, ObjectId
from config.mongodb import mongo

def my_ping_service():
    return 'ping', 200
