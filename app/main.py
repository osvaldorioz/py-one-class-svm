from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import one_class_svm
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/one-class-svm")
def calculo(points: int, nu: float, gamma: float):
    output_file = 'one-class-svm.png'
    # Generar datos de prueba (distribución normal y algunos puntos fuera de la distribución)
    data = torch.cat([torch.randn(points, 2), torch.randn(10, 2) + 5.0], dim=0)

    # Crear y entrenar el modelo One-Class SVM
    #nu = 0.1  # Proporción de anomalías esperadas
    #gamma = 0.5  # Parámetro del kernel RBF
    model = one_class_svm.OneClassSVM(nu, gamma)
    model.fit(data)

    # Predecir etiquetas (1 = normal, -1 = anomalía)
    labels = model.predict(data)

    # Separar los datos normales y las anomalías para graficar
    normal_data = data[torch.tensor(labels) == 1]
    anomalies = data[torch.tensor(labels) == -1]

    # Graficar los resultados
    plt.scatter(normal_data[:, 0], normal_data[:, 1], c='blue', label='Normal')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalías')
    plt.legend()    
    plt.title("Detección de Anomalías con One-Class SVM")
    #plt.show()
    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/one-class-svm-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)