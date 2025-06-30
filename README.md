# FA_TALLER2
FA II 2025-I: Prónosticos NN-RNN-CNN

Competencia Kaggle: https://www.kaggle.com/competitions/fa-ii-2025-i-pronosticos-nn-rnn-cnn

## Autores
- **Alexandra López Cuenca**  
- **Carlos Arturo Hoyos Rincón**

### Resumen de Archivos

#### **EDA.ipynb (Análisis Exploratorio de Datos)**
Este script realiza un análisis exploratorio exhaustivo de los datos de dengue, identificando patrones estacionales, tendencias, y la influencia de factores climáticos y ambientales. Destaca la necesidad de transformar los datos de dengue para manejar su asimetría y la importancia de filtrar años atípicos para mejorar la predicción de picos epidémicos. Concluye que la transformación logarítmica es crucial para capturar la magnitud de los picos y que la exclusión de ciertos años "ruidosos" mejora la señal para el entrenamiento del modelo.

#### **MLP.ipynb (Multi-Layer Perceptron)**
Este notebook implementa un modelo Perceptrón Multicapa (MLP) para la predicción de casos de dengue. Incluye la carga de datos, preprocesamiento (escalado), entrenamiento del modelo con optimización de hiperparámetros usando Optuna, y generación de predicciones para el año 2022. También visualiza las predicciones para los barrios con más casos.

#### **CNN.ipynb (Convolutional Neural Network)**
Este archivo presenta la implementación de una Red Neuronal Convolucional (CNN) para el análisis y pronóstico del dengue. El script abarca desde la carga y preprocesamiento de datos (escalado, creación de secuencias temporales) hasta la definición, entrenamiento y evaluación del modelo CNN. Utiliza Optuna para la optimización de hiperparámetros y genera predicciones para 2022. Además, compara la CNN con otras arquitecturas de redes neuronales.

#### **LSTM.ipynb (Long Short-Term Memory)**
Este notebook se centra en la aplicación de una red LSTM para la predicción de casos de dengue. Realiza un preprocesamiento de datos detallado, incluyendo escalado y preparación de secuencias temporales. El script define y entrena un modelo LSTM, utilizando Optuna para la optimización de hiperparámetros. Finalmente, genera y guarda las predicciones para el año 2022 y proporciona un resumen de los resultados y los archivos generados.

#### **GRU.ipynb (Gated Recurrent Unit)**
Este script implementa un modelo de Red Neuronal Recurrente con Unidades de Compuerta Recurrente (GRU) para la predicción de casos de dengue. Carga y preprocesa los datos, define la arquitectura de la red GRU, y utiliza Optuna para encontrar los mejores hiperparámetros. El modelo se entrena con los datos históricos y luego realiza predicciones iterativas para el año 2022, guardando los resultados en un archivo CSV.

#### **RNN_A01.ipynb (Recurrent Neural Network)**
Este notebook implementa un modelo de Red Neuronal Recurrente (SimpleRNN) para la predicción de casos de dengue. Incluye la importación de librerías, configuración de reproducibilidad, carga y preprocesamiento de datos (escalado y creación de secuencias), construcción y entrenamiento del modelo SimpleRNN. Finalmente, el script genera predicciones para el año 2022 y guarda el archivo de submission.

#### **TCN.ipynb (Temporal Convolutional Network)**
Este archivo desarrolla un modelo de Red Neuronal Convolucional Temporal (TCN) para la predicción del dengue. Cubre la carga de datos, preprocesamiento (escalado, creación de secuencias temporales), definición de la arquitectura TCN y entrenamiento. Emplea Optuna para la optimización de hiperparámetros y genera predicciones para 2022, incluyendo visualizaciones de los resultados.

#### **CNN_LSTM_A16.ipynb (Ensemble CNN y LSTM)**
Este notebook combina las arquitecturas CNN y LSTM para un modelo de ensamble robusto para la predicción de casos de dengue. Realiza una importación exhaustiva de librerías, configuración de semillas para reproducibilidad y definición de constantes clave. El script se enfoca en la preparación de datos y el ensamble de ambos modelos para generar predicciones finales para el año 2022, las cuales son guardadas en un archivo CSV.
