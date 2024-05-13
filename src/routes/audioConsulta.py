from flask import Blueprint

from services.audioConsulta import create_audioConsulta_service, getAll_audioConsulta_service, get_audioConsulta_service, update_audioConsulta_service, delete_audioConsulta_service

audioConsulta = Blueprint('audioConsulta', __name__)

@audioConsulta.route('/', methods=['GET'])
def getAll_audioConsulta():
    return getAll_audioConsulta_service()

@audioConsulta.route('/<id>', methods=['GET'])
def get_audioConsulta(id):
    return get_audioConsulta_service(id)

@audioConsulta.route('/', methods=['POST'])
def create_audioConsulta():
    return create_audioConsulta_service()

@audioConsulta.route('/<id>', methods=['PUT'])
def update_audioConsulta(id):
    return update_audioConsulta_service(id)

@audioConsulta.route('/<id>', methods=['DELETE'])
def delete_audioConsulta(id):
    return delete_audioConsulta_service(id)