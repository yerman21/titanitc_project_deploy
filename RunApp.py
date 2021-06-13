"""
@author: Yerman7@socialdata-peru.com
"""
from flask import Flask
from flask import request
from flask import render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

modelo = load('modelo_titanic.joblib')# Cargamos el modelo
encoder = load('label_encoder.joblib')# Cargarmo el label encoder

@app.route('/', methods=["GET"])
def main():
    return render_template('index.html')
    
@app.route('/predecir', methods=["POST"])
def predecir():
    datos = request.json
    print(datos)
    message = titanicPred(datos["input_clase"], datos["input_sexo"], datos["input_edad"])

    return message

def titanicPred(input_clase, input_sexo, input_edad):
    newEntry = {
            'Pclass': [input_clase], 
            'Sex': [input_sexo],
            'Age': [input_edad], 
            }
    newEntry = pd.DataFrame(newEntry)
    newEntry["Sex"] = encoder.transform(newEntry["Sex"])
    
    prediccion = modelo.predict(newEntry)[0]
    return 'La prediccion es '+("SOBREVIVIO" if prediccion == 1 else "MURIO")

if __name__ == '__main__':
    app.run(debug=True)