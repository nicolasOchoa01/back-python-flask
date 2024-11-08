from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import gridfs

from config.mongodb import mongo
from routes.audio_consulta_router import audio_consulta_service, init_audio_consulta_service
from routes.ping_router import ping_service

load_dotenv()

app = Flask(__name__)

@app.before_request
def before_request():
    headers = {'Access-Control-Allow-Origin': '*',
               'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
               'Access-Control-Allow-Headers': 'Content-Type'}
    if request.method.lower() == 'options':
        return jsonify(headers), 200

CORS(app, resources={r"/api/audioConsulta/*": {"origins": "http://localhost:4200"}})

app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo.init_app(app)

uri = os.getenv('MONGO_URI')
mongo_client = MongoClient(uri)
_mongod = mongo_client['AudioData']
_gridfs = gridfs.GridFS(_mongod)

# Inicializar el servicio de carga de audio
init_audio_consulta_service(_mongod, _gridfs)

app.register_blueprint(audio_consulta_service, url_prefix='/api/audioConsulta')
app.register_blueprint(ping_service, url_prefix='/api/ping')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=4000)
