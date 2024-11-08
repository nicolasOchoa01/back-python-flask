import base64
from datetime import datetime
import io
import os
import subprocess
import tempfile
from flask import jsonify, request, Response, send_file
from bson import Binary, json_util, ObjectId
import openai
from config.mongodb import mongo
from pydub import AudioSegment
import speech_recognition as sr
# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent

from functools import wraps
import random
from scipy.io import wavfile
from scipy.signal import resample
import re
import numpy as np
import jwt
from speechbrain.inference import SpeakerRecognition
import torchaudio
import datetime


torchaudio.set_audio_backend("soundfile")

# cargar el modelo preentrenado de SpeechBrain
modelo = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="src/pretrained_model", use_auth_token=False)


openai.api_key = os.getenv('openai.api_key')

oracion = " "
username = " "


def register_user_service(_mongod, _gridfs):
    try:
        # Obtener nombre de usuario y audio.wav que servira de contrasenia
        username = request.form['username']
        audio = request.files['password']

        # Verificar si el usuario ya existe
        if _mongod.db.users.find_one({"username": username}):
            return jsonify({
                "mensaje": "ya existe un usuario registrado con ese nombre en el sistema",
                "resultado": False}), 400

        # Subir el archivo de audio a GridFS y retener su ObjectId
        file_id = _gridfs.put(audio, filename=audio.filename)

        # Guardar los datos del usuario y la referencia al archivo en MongoDB
        user_data = {
            "username": username,
            "password_audio_file_id": file_id,
            "audios":[],
            "transcripciones":[]
        }
        _mongod.db.users.insert_one(user_data)

        return jsonify({
            "mensaje": "Usuario registrado con éxito",
            "resultado": True,
            "file_id": str(file_id)}), 200
    except Exception as e:
        return jsonify({'error al registrarse': str(e)}), 500

def loggin_username_service(_mongod):
    try:
        global oracion
        global username

        # obtiene el nombre de usuario enviado desde el front y busca en la base de datos y exite el usuario
        # si el usuario no exite, devuelve un mensaje de error
        # si el ususario exite, genera una oracion y la envia al front para que el usuario la lea
        username = request.form['username']
        if not _mongod.db.users.find_one({"username": username}):
            return jsonify({
                "mensaje":"no existe un usuario registrado con ese nombre en el sistema",
                "resultado": False}), 400
        oracion = generar_oracion()

        return jsonify({"mensaje": "existe el usuario, ahora lea la oracion generada",
                        "resultado": True,
                        "oracion": oracion}), 200
    except Exception as e:
        return jsonify({'error al valir el usuario': str(e)}), 500

def loggin_password_service(_mongod, _gridfs):
  
        global oracion
        global username

        # obtener el audio desde el front en bytes
        password_front = request.files['password']
        password_front_bytes = password_front.read()

        # procesar el audio para convertirlo del tipo int16 de un canal, retorna el audio en bytes formateado
        password_front_bytes_processed = procesar_audio(password_front_bytes)

        # crea un archivo temporal para almacenar el audio del front y obtiene su ruta
        ruta_pass_front = convertir_audio_temporal(password_front_bytes_processed)

        # realiza la transcripcion y la valida, obtiene un valor booleano
        # transcripcion = transcribir(ruta_pass_front)
        transcripcion = oracion
        result_lectura = validar_lectura(transcripcion, oracion)

        # si la validacion da False, devuelve un mensaje de error y una nueva oracion para leer
        if not result_lectura:
            eliminar_archivo_temporal(ruta_pass_front)
            oracion = generar_oracion()
            return jsonify({
                "mensaje":"la lectura no coincide con la oracion generada",
                "resultado": False,
                "oracion": oracion,
                "transcripcion": transcripcion}), 400

        # obtiene el audio registrado en grid en bytes
        password_db_bytes = get_password_audio_db(_mongod, _gridfs, username)

        # procesar el audio al tipo int16 de un canal
        password_db_bytes_processed = procesar_audio(password_db_bytes)

        # crea un archivo temporal para almacenar el audio registrado en grid y devuelve su ruta
        ruta_pass_db = convertir_audio_temporal(password_db_bytes_processed)

        # valida la voz del hablante con la del usuario registrado, obtiene un valor booleano
        result_voz = validar_voz(ruta_pass_front , ruta_pass_db)
        # result_voz = True

        # si la validacion es False, devuelve un mensaje de error y una nueva oracion generada para leer
        if not result_voz:
            eliminar_archivo_temporal(ruta_pass_front)
            eliminar_archivo_temporal(ruta_pass_db)
            oracion = generar_oracion()
            return jsonify({
                "mensaje":"la voz no coincide con la del usuario ingresado",
                "resultado": False,
                "oracion": oracion}), 400

        token = generar_token(username)
        # la validacion se completa, envia mensaje y codigo de estado ok, elimina archivos temporales
        eliminar_archivo_temporal(ruta_pass_front)
        eliminar_archivo_temporal(ruta_pass_db)
        return jsonify({
                "mensaje":"felicidades te loggeaste con exito",
                "resultado": True,
                "oracion": oracion,
                "transcripcion": transcripcion,
                "token": token}), 200



# subir un informe al sistema
def upload_transcripcion_service(_mongod, username):
    try:
        data = request.get_json()
        if "transcripcion" not in data:
            return jsonify({'error': 'No se ha enviado el archivo'}), 400

        # Obtener el informe
        transcripcion = data["transcripcion"]
        
        # guardarlo en mongo
        _mongod.db.users.update_one(
            {"username": username},
            {"$push": {"transcripciones": transcripcion}}
        )
        return jsonify({'mensaje': 'transcripcion subida correctamente',
                        "resultado": True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# def is_audio_silent(file_path, silence_threshold=-0.0, silence_duration=5000):
#     """
#     Verifica si un archivo de audio está en silencio.

#     :param file_path: Ruta del archivo de audio.
#     :param silence_threshold: Umbral de silencio en dB (por defecto -50.0).
#     :param silence_duration: Duración mínima del silencio en milisegundos (por defecto 1000 ms).
#     :return: True si el audio está en silencio, False si contiene sonido.
#     """
#     audio = AudioSegment.from_file(file_path)
#     audio = audio.normalize()

#     # Ajustar el umbral de silencio
#     silence_thresh = audio.dBFS + silence_threshold
#     print(f"Nivel de silencio: {silence_thresh} dB")
#     print(f"Nivel promedio del audio: {audio.dBFS} dB")

#     # thresh = segment.dBFS - (segment.max_dBFS - segment.dBFS)
#     # non_silent_ranges = pydub.silence.detect_nonsilent(segment, min_silence_len=1000, silence_thresh=thresh)

#     # Utiliza el método detect_nonsilent para encontrar rangos de audio no silencioso
#     nonsilent_ranges = detect_nonsilent(
#         audio,
#         min_silence_len=silence_duration,
#         # silence_thresh=audio.dBFS + silence_threshold
#         silence_thresh=silence_thresh + silence_threshold
#     )

#      # Imprimir los rangos no silenciosos detectados
#     if nonsilent_ranges:
#         print("Rangos no silenciosos detectados:", nonsilent_ranges)
#         return False  # Hay audio, por lo tanto, no es silencio
#     else:
#         print("No se detectaron rangos no silenciosos. El audio es considerado silencio.")
#         return True  # No hay audio, se considera silencio

def is_audio_silent(file_path):
    """
    Verifica si un archivo de audio contiene habla o solo ruido/silencio.

    :param file_path: Ruta del archivo de audio.
    :return: True si el audio contiene solo ruido o silencio, False si contiene habla.
    """
    recognizer = sr.Recognizer()
    audio_segment = AudioSegment.from_file(file_path)

    # Convertir el archivo de audio a un formato compatible para el análisis
    wav_path = file_path.replace(".wav", "_temp.wav")
    audio_segment.export(wav_path, format="wav")

    try:
        # Cargar el archivo de audio para el reconocimiento de voz
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

            try:
                # Intentar reconocer el habla en el archivo de audio
                recognizer.recognize_google(audio_data)
                print("Se detectó habla, no es silencio ni solo ruido")
                return False  # Se detectó habla, no es silencio ni solo ruido
            except sr.UnknownValueError:
                print("No se detectó habla, se considera ruido o silencio")
                return True  # No se detectó habla, se considera ruido o silencio
            except sr.RequestError:
                print("Error con el servicio de reconocimiento de voz")
                return True
    finally:
        # Eliminar archivo temporal después de que todos los procesos hayan terminado
        if os.path.exists(wav_path):
            os.remove(wav_path)

def upload_audio_file_in_storage_service(_mongod, _gridfs, username):
    try:
        if 'archivo' not in request.files:
            return jsonify({'error': 'No se ha enviado el archivo'}), 400

        # Obtener el archivo enviado en el formulario
        archivo = request.files['archivo']

        # Crear un archivo temporal para verificar el silencio
        temp_file_path = create_unique_temp_file(suffix=".wav")

        # Guardar el archivo en un archivo temporal para análisis
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(archivo.read())
            temp_file.flush()


        # Verificar si el audio contiene solo ruido o silencio
        if is_audio_silent(temp_file_path):
            os.remove(temp_file_path)
            return jsonify({'error': 'El audio no contiene habla y no se almacenará.'}), 412

        # Reiniciar el archivo en el objeto de archivo (se requiere para cargar nuevamente)
        archivo.seek(0)

        title = request.form.get('title')
        full_filename = f"{title}"
        mimetype = archivo.mimetype
        file_binary = Binary(archivo.read())
        size = len(file_binary)

        # Subir el archivo a GridFS
        file_id = _gridfs.put(file_binary, filename=full_filename, content_type=mimetype)

        _mongod.db.users.update_one(
            {"username": username},          # Filtro para encontrar el documento del usuario
            {"$push": {"audios": file_id}}  # Agregar el nuevo ObjectId a la lista 'audios'
        )


        # Eliminar el archivo temporal
        os.remove(temp_file_path)

        # Responder con el ID del archivo subido y los metadatos
        return jsonify({
            'message': 'Archivo subido correctamente',
            'file_id': str(file_id),
            'metadata': {
                'filename': full_filename,
                'mimetype': mimetype,
                'size': size,
                # 'sizeUnit': size_unit
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def delete_audio_file_from_storage_service(file_id, _gridfs):
    try:
        # Convertir el file_id a un ObjectId
        object_id = ObjectId(file_id)

        # Verificar si el archivo existe en GridFS
        if not _gridfs.exists(object_id):
            return jsonify({'error': 'El archivo no existe'}), 404

        # Eliminar el archivo de GridFS
        _gridfs.delete(object_id)

        # Responder con un mensaje de confirmación
        return jsonify({'message': 'Archivo eliminado correctamente'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_audio_from_gridfs(file_id, _gridfs):
    try:
        # Intenta encontrar el archivo en GridFS
        file_data = _gridfs.find_one({"_id": ObjectId(file_id)})

        if not file_data:
            return None, 'Archivo no encontrado'

        # Recuperar el archivo
        file_binary = _gridfs.get(file_data._id).read()
        mimetype = file_data.content_type
        filename = file_data.filename

        return file_binary, mimetype, filename

    except Exception as e:
        return None, str(e)


def get_audio_file(file_id, _gridfs):
    try:
        file_binary, mimetype, filename = get_audio_from_gridfs(file_id, _gridfs)

        if file_binary is None:
            return jsonify({'error': mimetype}), 404  # mimetype contiene el mensaje de error

        # Verifica que el mimetype sea correcto para un archivo de audio
        if not mimetype.startswith('audio/'):
            return jsonify({'error': 'El archivo recuperado no es de tipo audio'}), 400

        # Enviar el archivo recuperado como descarga
        return send_file(io.BytesIO(file_binary), mimetype=mimetype, as_attachment=True, download_name=filename)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_unique_temp_file(suffix=".wav"):
    """
    Crea un archivo temporal con un nombre único y devuelve el nombre del archivo.
    El archivo no se eliminará automáticamente al cerrar para permitir su manipulación posterior.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()  # Cerramos el archivo para poder usarlo en otros procesos
    return temp_file.name


def send_audio_to_whisper_service(file_id, _gridfs):
    try:
        # Usa la función get_audio_from_gridfs para obtener el archivo desde GridFS
        file_binary, mimetype, filename = get_audio_from_gridfs(file_id, _gridfs)

        if file_binary is None:
            return jsonify({'error': mimetype}), 404  # mimetype contiene el mensaje de error

        # Verifica que el archivo sea de tipo audio
        if not mimetype.startswith('audio/'):
            return jsonify({'error': 'El archivo recuperado no es de tipo audio'}), 400




        # # Crear archivos temporales con nombres únicos                                  **************************
        # temp_file_path = create_unique_temp_file(suffix=".wav")                                modificado
        # converted_file_path = create_unique_temp_file(suffix=".wav")                    **************************


        converted_file_path = convertir_audio_temporal(file_binary)

        # # Guardar el archivo binario en el archivo temporal
        # with open(temp_file_path, 'wb') as temp_file:
        #     temp_file.write(file_binary)
        #     temp_file.flush()

        try:
            # Guardar y convertir el archivo a .wav                                       **************************
            # save_file(temp_file_path, file_binary)                                               modificado
            # convert_to_wav(temp_file_path, converted_file_path)                         **************************

            # # Convertir el archivo a formato .wav usando ffmpeg
            # subprocess.run(['ffmpeg', '-y', '-i', temp_file_path, converted_file_path], check=True)

            # Transcribir el audio usando Whisper
            transcript_text = transcribe_audio(converted_file_path)

            # # Ahora enviamos el archivo convertido a Whisper para transcripción
            # with open(converted_file_path, 'rb') as audio_file:
            #     transcription  = openai.audio.transcriptions.create(
            #         model="whisper-1",
            #         file=audio_file
            #     )

            # La transcripción se obtiene accediendo al atributo "text"
            # transcript_text = transcription.text

            # # Eliminar los archivos temporales
            # os.remove(temp_file_path)
            # os.remove(converted_file_path)

            # # return jsonify({'transcription': transcript_text}), 200
            # return generate_report_from_transcript(transcript_text)

            # Verifica si el texto de la transcripción está vacío o solo contiene espacios en blanco
            if not transcript_text or transcript_text.strip() == "":
                return jsonify({'error': 'La transcripción está vacía. No se puede generar un reporte.'}), 412

            # Generar el reporte formateado a partir de la transcripción
            return generate_report_from_transcript(transcript_text)

        finally:
            # Asegurar la eliminación de archivos temporales
            # cleanup_files([temp_file_path, converted_file_path])                         ************  modificado ************
            eliminar_archivo_temporal(converted_file_path)

    except Exception as e:
        return jsonify({'error': f'Error en el procesamiento del audio: {str(e)}'}), 500


def save_file(file_path, file_binary):
    # Guardar el archivo binario en el archivo temporal
    with open(file_path, 'wb') as temp_file:
        temp_file.write(file_binary)
        temp_file.flush()

# def convert_to_wav(input_path, output_path):                                                    **************************
#     """Convierte un archivo de audio a .wav usando ffmpeg."""                                        funcion comentada
#     try:                                                                                        **************************
#         subprocess.run(['ffmpeg', '-y', '-i', input_path, output_path], check=True)
#     except subprocess.CalledProcessError as e:
#         raise Exception(f"Error en la conversión a WAV: {str(e)}")

def transcribe_audio(file_path):
    """Usa el modelo de Whisper para transcribir el audio."""
    try:
        # Ahora enviamos el archivo convertido a Whisper para transcripción
        with open(file_path, 'rb') as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        raise Exception(f"Error en la transcripción del audio: {str(e)}")


def cleanup_files(file_paths):
    """Elimina los archivos temporales especificados."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Error al eliminar el archivo {path}: {str(e)}")


def generate_report_from_transcript(transcript_text):
    """
    Función que toma una transcripción y la envía a la API de OpenAI para formatearla en un reporte.
    Genera un reporte médico formateado a partir de una transcripción.

    :param transcript_text: Texto de la transcripción obtenida de Whisper.
    :return: Reporte formateado.
    """

    # Agrego un log para verificar el contenido de transcript_text
    print(f"Contenido de transcript_text: '{transcript_text}'")  # Debugging

    # Instrucciones y transcripción para enviar a ChatGPT
    messages = [
        {"role": "system", "content": "Eres un asistente que formatea transcripciones en un reporte médico estructurado."},
        {"role": "user", "content": f"""
        Por favor, genera un reporte médico estructurado a partir de la siguiente transcripción. Usa el siguiente formato:
        Título: [El título general del reporte]
        Subtítulo 1: [Descripción breve]
        Texto: [Cuerpo de la sección]
        Subtítulo 2: [Descripción breve]
        Texto: [Cuerpo de la sección]
        Conclusión: [Descripción breve]
        Texto: [Conclusión]

        Transcripción:
        {transcript_text}
        """}
    ]
    try:
        # Llamada a la API de OpenAI para generar el reporte
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "Eres un asistente que da formato técnico a reportes médico."},
        #         {"role": "user", "content": prompt}
        #     ]
        # )

        # Llamada a la API de OpenAI para generar el reporte
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        # Obtener el reporte formateado desde la respuesta
        formatted_report = response.choices[0].message.content

        # Retornar el reporte formateado
        return jsonify({'formatted_report': formatted_report}), 200

    except Exception as e:
         return jsonify({'error': f'Error al generar el reporte: {str(e)}'}), 500










def save_audio_file_in_storage_service(request):
    file = request.files['archivo']
    title_audio = request.form.get('title', '')

    file_name = file.filename
    type = file.mimetype
    file_binary = Binary(file.read())
    date_time = datetime.now()
    size = len(file_binary)

    response = mongo.db.audioConsulta.insert_one({
        'file_name': file_name,
        'type': type,
        'file_binary': file_binary,
        'date_time': date_time,
        'size': size,
        'stored_in_db': False,
        'title': title_audio
    })
    result = {
        'id': str(response.inserted_id),
        'title': title_audio,
        'type': type,
        'date_time': date_time,
        'size': size,
        'stored_in_db': False
    }
    return result

def create_audioConsulta_service():
    result_a = save_audio_file_in_storage_service(request)
    result_b = send_audio_to_whisper_service(request)
    # aca recuperar archivo y pasarlo a nueva function
    # que lo convierta a .wav y lo envie a whisper para
    # obtener repuesta
    # despues sigue el procesamiento de esta function para
    # guardar el audio y transcripcion en mongo

    if result_a and result_b:
        return jsonify(result_b), 200
    else:
        return 'System internal Error', 400

# def get_all_audio_service():
#     data = mongo.db.audioConsulta.find()
#     result = json_util.dumps(data)
#     return Response(result, mimetype='application/json')

# def  get_audio_service(id):
#     data = mongo.db.audioConsulta.find_one({'_id': ObjectId(id)})
#     result = json_util.dumps(data)
#     return Response(result, mimetype='application/json')

# def update_audio_service(id):
#     data = request.get_json()
#     if len(data) == 0:
#         return 'Invalid payload', 400

#     response = mongo.db.audioConsulta.update_one({'_id': ObjectId(id)},{'$set': data})

#     if response.modified_count >= 1:
#         return 'audioConsulta updated successfully', 200
#     else:
#         return 'audioConsulta not found', 404

# def delete_audio_service(id):
#     response = mongo.db.audioConsulta.delete_one({'_id': ObjectId(id)})
#     if response.deleted_count >= 1:
#         return 'audioConsulta deleted successfully', 200
#     else:
#         return 'audioConsulta not found', 404

# def delete_audio_service(id):
#      response = mongo.db.audioConsulta.delete_one({'_id': ObjectId(id)})
#      if response.deleted_count >= 1:
#          return 'audioConsulta deleted successfully', 200
#      else:
#          return 'audioConsulta not found', 404







def generar_token(username):
    # Crear el JWT
    token = jwt.encode({
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Expiración en 1 hora
    }, 'secretkey', algorithm="HS256")
    return token

def get_password_audio_db(_mongod, _gridfs, username):
    # Buscar el usuario
    user = _mongod.db.users.find_one({"username": username})
    
    if not user:
        return jsonify({"mensaje": "Usuario no encontrado"}), 404

    # Obtener el archivo de audio desde GridFS
    file_id = user["password_audio_file_id"]
    audio_file = _gridfs.get(file_id)

    # Devolver el archivo de audio
    return audio_file.read()

def generar_oracion():
    sujeto = random.choice(sujetos)
    verbo = random.choice(verbos)
    complemento = random.choice(complementos)
    texto = sujeto + " " + verbo + " " + complemento
    return texto

# toma un audio en bytes, crea un archivo temporal y lo convierte en wav. devuelve la ruta del archivo temporal
def convertir_audio_temporal(audio):
    try:
        # Crear archivos temporales con nombres únicos
        converted_file_path = create_unique_temp_file(suffix=".wav")

        sample_rate, file_binary = wavfile.read(io.BytesIO(audio))
        wavfile.write(converted_file_path, sample_rate, file_binary)

        return converted_file_path
    except Exception as e:
        return jsonify({'error al convertir audio temporal': str(e)}), 500


def eliminar_archivo_temporal(ruta):
    os.remove(ruta)

# Convertir a minúsculas y eliminar puntuaciones, caracteres especiales y espacios
def clear_text(text):
    clean_text = re.sub(r'[^a-zA]', '', text.lower())
    return clean_text

def validar_lectura(transcripcion, texto):
    transcripcion = clear_text(transcripcion)
    texto = clear_text(texto)
    if transcripcion == texto:
        return True
    else:
        return False


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            print('token faltante')
            return jsonify({'mensaje': 'Token faltante'}), 403
        try:
            token = token.split(" ")[1]
            data = jwt.decode(token, 'secretkey', algorithms=["HS256"])
            username = data['username']
        except jwt.ExpiredSignatureError:
            print('token expirado')
            return jsonify({'mensaje': 'Token expirado'}), 403
        except jwt.InvalidTokenError:
            print('token invalido')
            return jsonify({'mensaje': 'Token inválido'}), 403

        return f(username, *args, **kwargs)
    return decorated


def transcribir(ruta):
    # Enviar el archivo a Whisper para la transcripción
    with open(ruta, 'rb') as audio:
        transcripcion  = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    # Obtener el texto transcrito
    transcripcion_text = transcripcion.text
    # print("Transcripción:", transcripcion_text)
    return transcripcion_text

def validar_voz(ruta_audio1, ruta_audio2):

    señal1, fs1 = torchaudio.load(ruta_audio1)
    señal2, fs2 = torchaudio.load(ruta_audio2)
    # verificar si las dos voces coinciden
    score, prediccion = modelo.verify_batch(señal1, señal2)

    # resultado de la verificación, valor booleano
    return prediccion

# toma un audio wav con cualquier configuracion y retorna el audio en bytes con un canal y del tipo int16
def procesar_audio(file_bytes):
    try:
        file = io.BytesIO(file_bytes)
        # Leer el archivo de audio desde el archivo subido
        fs, data = wavfile.read(file)
        
        # Convertir a mono si es estéreo
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        
        # Cambiar la frecuencia de muestreo a 16000 Hz si es diferente
        target_fs = 16000
        if fs != target_fs:
            num_samples = int(len(data) * target_fs / fs)
            data = resample(data, num_samples)
        
        # Asegurarse de que el audio esté en formato int16
        data = np.asarray(data, dtype=np.int16)

        # Crear un buffer en memoria para almacenar el audio en bytes
        audio_bytes = io.BytesIO()
        wavfile.write(audio_bytes, target_fs, data)
        audio_bytes.seek(0)  # Volver al inicio del buffer

        return audio_bytes.read()  # Retornar el contenido en bytes
    except Exception as e:
        return jsonify({'error al procesar el audio': str(e)}), 500

# Lista de 100 sujetos
sujetos = [
    "El gato", "La casa", "Un coche", "Una persona", "El perro", "El pájaro", "El niño", "La niña",
    "El profesor", "El estudiante", "El árbol", "La montaña", "El río", "El mar", "El viento",
    "El soldado", "El piloto", "El doctor", "El robot", "El astronauta", "La madre", "El padre",
    "El hermano", "La hermana", "El músico", "El pintor", "El actor", "El escritor", "El ingeniero",
    "El chef", "El guardia", "El policía", "El bombero", "El dragón", "El león", "La mariposa",
    "El tiburón", "El lobo", "El oso", "El ciervo", "El ratón", "El científico", "El investigador",
    "El vecino", "El abogado", "El periodista", "El carpintero", "El mecánico", "El electricista",
    "El granjero", "El pescador", "El cazador", "El ciclista", "El corredor", "El nadador", "El pintor",
    "El escritor", "El explorador", "El aventurero", "El viajero", "El turista", "El guía", "El mago",
    "El rey", "La reina", "El príncipe", "La princesa", "El fantasma", "El vampiro", "El monstruo",
    "El alienígena", "El robot", "El samurái", "El ninja", "El pirata", "El guerrero", "El caballero",
    "El capitán", "El director", "El entrenador", "El jugador", "El bailarín", "El cantante",
    "El músico", "El artista", "El fotógrafo", "El cineasta", "El agricultor", "El tendero",
    "El comerciante", "El banquero", "El administrador", "El político", "El filósofo", "El matemático",
    "El físico", "El químico", "El biólogo", "El arquitecto", "El diseñador", "El programador",
    "El desarrollador", "El técnico", "El operador", "El conductor"
]

# Lista de 100 verbos
verbos = [
    "come", "corre", "salta", "mira", "duerme", "camina", "canta", "baila", "conduce", "nada",
    "vuela", "habla", "escribe", "lee", "juega", "construye", "destruye", "crea", "dibuja", "pinta",
    "explora", "descubre", "investiga", "compra", "vende", "prepara", "lava", "seca", "friega",
    "arregla", "rompe", "abre", "cierra", "enciende", "apaga", "atrapa", "lucha", "cocina", "hornea",
    "traduce", "calcula", "enseña", "aprende", "explica", "colorea", "graba", "escucha", "ve", "observa",
    "detecta", "analiza", "programa", "prueba", "mejora", "crece", "disminuye", "calcula", "esculpe",
    "compone", "interpreta", "crea", "borra", "mueve", "gira", "cae", "se levanta", "gana", "pierde",
    "celebra", "descansa", "invita", "recibe", "viaja", "explora", "descubre", "coloca", "empaqueta",
    "envuelve", "abre", "desempaca", "examina", "analiza", "distribuye", "clasifica", "escoge",
    "selecciona", "envía", "recoge", "saca", "mete", "suelta", "agarra", "espera", "reúne", "separa",
    "conecta", "desconecta", "arma", "desarma", "organiza", "resuelve", "atrapa", "escapa", "protege"
]

# Lista de 100 complementos
complementos = [
    "rápidamente", "en el parque", "bajo la lluvia", "con cuidado", "en silencio", "sin hacer ruido",
    "con alegría", "en la biblioteca", "en la montaña", "en la ciudad", "en el desierto", "en el campo",
    "en el mar", "en el bosque", "en el río", "bajo el sol", "en la sombra", "en el avión", "en el tren",
    "en el coche", "en la bicicleta", "en la playa", "en el estadio", "en el teatro", "en el museo",
    "en la galería", "en la tienda", "en la escuela", "en la universidad", "en el hospital", "en el laboratorio",
    "en la oficina", "en la fábrica", "en el mercado", "en el restaurante", "en el café", "en el bar",
    "en la plaza", "en el jardín", "en la piscina", "en la cancha", "en el gimnasio", "en el zoológico",
    "en la farmacia", "en la estación", "en el aeropuerto", "en la base", "en el puerto", "en el barco",
    "en el submarino", "en la nave espacial", "en la casa", "en el apartamento", "en el edificio",
    "en el rascacielos", "en la cueva", "en la mina", "en la torre", "en el castillo", "en la fortaleza",
    "en la iglesia", "en la catedral", "en el templo", "en la mezquita", "en el mercado", "en el supermercado",
    "en la tienda de ropa", "en la librería", "en la ferretería", "en el banco", "en la peluquería",
    "en la estación de tren", "en la parada de autobús", "en la autopista", "en el túnel", "en el puente",
    "en el parque de atracciones", "en el circo", "en el concierto", "en la conferencia", "en el festival",
    "en la exposición", "en el evento", "en la feria", "en el mercado de pulgas", "en la tienda de comestibles",
    "en la oficina de correos", "en el centro comercial", "en el salón de belleza", "en el sofá", "en el club",
    "en la discoteca", "en la heladería", "en la pastelería", "en la panadería", "en la carnicería",
    "en la pescadería", "en la floristería", "en la joyería", "en la tienda de electrónica", "en la gasolinera"
]