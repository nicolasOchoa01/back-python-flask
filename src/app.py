from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os

from config.mongodb import mongo
from routes.audioConsulta import audioConsulta

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/audioConsulta/*": {"origins": "http://localhost:4200"}})

app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

app.register_blueprint(audioConsulta, url_prefix='/audioConsulta')

if __name__ == '__main__':
    app.run(debug=True, port=4000)