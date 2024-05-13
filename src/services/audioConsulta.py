from flask import request, Response 
from bson import json_util, ObjectId
from config.mongodb import mongo

def create_audioConsulta_service():
    data = request.get_json()
    blob = data.get('blob', None)
    title = data.get('title', None)
    fecha = datetime.now()

    if blob and title:

        archivo_binario = Binary(blob)

        tipo = blob.get('type', None)
        peso = blob.get('size', None)

        response = mongo.db.audioConsulta.insert_one({
            'nombre': title,
            'tipo': tipo,
            'archivo': archivo_binario,
            'fecha': fecha,
            'peso': peso,
            'done': False
        })
        result = {
            'id': str(response.inserted_id),
            'nombre': title,
            'tipo': tipo,
            'archivo': {
                'blob': blob,
                'title': title
            },
            'fecha': fecha,
            'peso': peso,
            'done': False
        }
        return jsonify(result), 200
    else:
        return 'Invalid payload', 400
        
def getAll_audioConsulta_service():
    data = mongo.db.audioConsulta.find()
    result = json_util.dumps(data)
    return Response(result, mimetype='application/json')

def  get_audioConsulta_service(id):
    data = mongo.db.audioConsulta.find_one({'_id': ObjectId(id)})
    result = json_util.dumps(data)
    return Response(result, mimetype='application/json')

def update_audioConsulta_service(id):
    data = request.get_json()
    if len(data) == 0:
        return 'Invalid payload', 400
    
    response = mongo.db.audioConsulta.update_one({'_id': ObjectId(id)},{'$set': data})

    if response.modified_count >= 1:
        return 'audioConsulta updated successfully', 200
    else:
        return 'audioConsulta not found', 404

def delete_audioConsulta_service(id):
    response = mongo.db.audioConsulta.delete_one({'_id': ObjectId(id)})
    if response.deleted_count >= 1:
        return 'audioConsulta deleted successfully', 200
    else:
        return 'audioConsulta not found', 404


    
