from flask import Blueprint, request

from services.audio_consulta_service import register_user_service, loggin_password_service, loggin_username_service, token_required, upload_transcripcion_service
from services.audio_consulta_service import send_audio_to_whisper_service, upload_audio_file_in_storage_service, get_audio_file, delete_audio_file_from_storage_service
#create_audio_service, get_all_audio_service, get_audio_service, update_audio_service, delete_audio_service,

audio_consulta_service = Blueprint('audio_consulta_service', __name__)

def init_audio_consulta_service(db, fs):
    # @audio_consulta_service.route('/', methods=['GET'])
    # def get_all_audio():
    #     return get_all_audio_service()

    # @audio_consulta_service.route('/<id>', methods=['GET'])
    # def get_audio(id):
    #     return get_audio_service(id)

    @audio_consulta_service.route('/register', methods=['POST'])
    def register_user():
        return register_user_service(db, fs)

    @audio_consulta_service.route('/loggin-username', methods=['POST'])
    def loggin_username():
        return loggin_username_service(db)

    @audio_consulta_service.route('/loggin-password', methods=['POST'])
    def loggin_password():
        return loggin_password_service(db, fs)



    @audio_consulta_service.route('/transcribe', methods=['POST'])
    @token_required
    def transcribe_audio_file(username):
        data = request.get_json()
        file_id = data['file_id']
        # return send_audio_to_whisper_service('66fd9f9a5eaa8d65a86ad172', fs)
        return send_audio_to_whisper_service(file_id, fs)

    @audio_consulta_service.route('/upload', methods=['POST'])
    @token_required
    def upload_audio_file_in_storage(username):
        return upload_audio_file_in_storage_service(db, fs, username)
    
    @audio_consulta_service.route('/upload-transcripcion', methods=['POST'])
    @token_required
    def upload_transcripcion(username):
        return upload_transcripcion_service(db, username)

    @audio_consulta_service.route('/file/<file_id>', methods=['GET'])
    @token_required
    def retrieve_audio_file(file_id, username):
        return get_audio_file(file_id, fs)

    # @audio_consulta_service.route('/transcribe/<file_id>', methods=['POST'])
    # def send_audio_to_whisper(file_id):
    #     return send_audio_to_whisper_service(file_id, fs)

    # @audio_consulta_service.route('/<id>', methods=['PUT'])
    # def update_audio(id):
    #     return update_audio_service(id)

    # @audio_consulta_service.route('/<id>', methods=['DELETE'])
    # def delete_audio(id):
    #     return delete_audio_service(id)

    @audio_consulta_service.route('/delete/file/<file_id>', methods=['DELETE'])
    @token_required
    def delete_audio_file(file_id, username):
        return delete_audio_file_from_storage_service(file_id, fs)
