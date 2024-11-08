from flask import Blueprint

from services.ping_service import my_ping_service

ping_service = Blueprint('ping_service', __name__)

@ping_service.route('/', methods=['GET'])
def my_ping():
    return my_ping_service()
