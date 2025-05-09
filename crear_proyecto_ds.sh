#!/bin/bash

# Solicitar nombre del 
echo "Script para crear la estructura base de un proyecto de Ciencia de Datos"
read -p "Nombre del nuevo proyecto: " nombre

# Crear carpeta principal y moverse a ella
mkdir "$nombre"
cd "$nombre" || exit

# Crear estructura de carpetas
mkdir data data/processed data/raw notebooks src reports reports/figures models .vscode

# Crear archivos iniciales
echo "# Proyecto de Ciencia de Datos: $nombre" > README.md
echo "Este directorio contiene los datos en bruto o procesados." > data/README.md

# .gitignore típico
cat <<EOF > .gitignore
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
EOF

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar librerías comunes
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

# Guardar dependencias
pip freeze > requirements.txt

# Configuración para VS Code
cat <<EOF > .vscode/settings.json
{
  "python.defaultInterpreterPath": "venv/bin/python"
}
EOF

# Crear notebook de analisis exploratorio de datos
cat <<EOF > notebooks/EDA.ipynb
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

# Crear notebook de preprocesamiento de datos
cat <<EOF > notebooks/data_preprocessing.ipynb
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

# Crear notebook de entrenamiento de modelos
cat <<EOF > notebooks/model_training.ipynb
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

# Crear notebook de evaluación de modelos
cat <<EOF > notebooks/model_evaluation.ipynb
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

echo "AVISO: En el archivo de configuración config.py debes completar la ruta de RAW_DATA_FILE con el nombre del archivo de datos en bruto que vas a usar."
# Crear script de configuracion base
cat <<EOF > src/config.py
"""
Configuración para el proyecto $nombre
Contiene constantes, rutas y parámetros de configuración.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Rutas de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_REPORT_DIR = os.path.join(REPORTS_DIR, 'models')
EVALUATION_DIR = os.path.join(REPORTS_DIR, 'evaluation')

# Rutas de archivos
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, '')
X_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_train.csv')
X_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_test.csv')
Y_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')
Y_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, 'standard_scaler.pkl')
BEST_PARAMS_FILE = os.path.join(MODELS_REPORT_DIR, 'best_params.json')
OPTIMAL_THRESHOLD_FILE = os.path.join(MODELS_REPORT_DIR, 'optimal_threshold.json')
MODEL_COMPARISON_FILE = os.path.join(EVALUATION_DIR, 'model_comparison.csv')
THRESHOLD_SENSITIVITY_FILE = os.path.join(EVALUATION_DIR, 'threshold_sensitivity.csv')
OPTIMAL_THRESHOLDS_FILE = os.path.join(EVALUATION_DIR, 'optimal_thresholds.json')
FALSE_POSITIVES_FILE = os.path.join(EVALUATION_DIR, 'false_positives.csv')
FALSE_NEGATIVES_FILE = os.path.join(EVALUATION_DIR, 'false_negatives.csv')

# Parámetros generales
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5  # Umbral por defecto para clasificación

# Parámetros para tratamiento de outliers
OUTLIER_THRESHOLD = 1.5  # Para método IQR


# Configuración de visualización
def set_plot_style():
    """Configura el estilo de visualización para gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Paleta de colores personalizada
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#e67e22', '#7f8c8d', '#27ae60'
    ])

# Función para generar nombre de archivo con timestamp
def get_timestamped_filename(base_name, extension):
    """Genera un nombre de archivo con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

# Parámetros para grids de hiperparámetros
def get_param_grids():
    """Retorna los grids de parámetros para búsqueda de hiperparámetros."""
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, 5, 10]  # Para manejar desbalanceo
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6, -1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 1.0],
            'class_weight': [None, 'balanced']
        }
    }
    return param_grids


EOF

# Crear script de preparación de datos base
cat <<EOF > src/data_prep.py
"""
Funciones para la preparación y procesamiento de datos en el proyecto $nombre.
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union

from src.config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, SCALER_FILE,
    OUTLIER_THRESHOLD
)

from src.utils import timer_decorator

@timer_decorator
def load_data(file_path: str = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    print(f"Cargando datos desde {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    df = pd.read_csv(file_path)
    print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    return df

EOF

# Crear script de evaluación de datos base
cat <<EOF > src/evaluate.py
"""
Funciones para la evaluación de modelos en el proyecto $nombre.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Añade esta línea
import joblib
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import glob

EOF

# Crear script de generación de informes base
cat <<EOF > src/generate_report.py
"""
Funciones para generar informes de resultados de modelos.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

def generate_markdown_report(metrics_dict: Dict[str, Dict[str, Any]], 
                            best_params: Dict[str, Dict[str, Any]],
                            output_path: str):
    """
    Genera un informe resumido en formato Markdown.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        best_params: Diccionario con mejores parámetros por modelo
        output_path: Ruta donde guardar el informe
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Título
        f.write("# Informe de Evaluación de Modelos de Detección de Fraude\n\n")
        
        # Resumen de métricas
        f.write("## Resumen de Métricas\n\n")
        
        # Crear tabla de métricas
        f.write("| Modelo | Accuracy | Precision | Recall | F1 | F2 | AUC | AP |\n")
        f.write("|--------|----------|-----------|--------|----|----|-----|----|\n")
        
        for name, metrics in metrics_dict.items():
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            f2 = metrics.get('f2', 'N/A')
            auc = metrics.get('roc_auc', 'N/A')
            ap = metrics.get('avg_precision', 'N/A')
            
            if isinstance(accuracy, (int, float)):
                f.write(f"| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {f2:.4f} | {auc:.4f} | {ap:.4f} |\n")
            else:
                f.write(f"| {name} | {accuracy} | {precision} | {recall} | {f1} | {f2} | {auc} | {ap} |\n")
        
        f.write("\n")
        
        # Mejores parámetros
        f.write("## Mejores Hiperparámetros\n\n")
        
        for name, params in best_params.items():
            f.write(f"### {name}\n\n")
            f.write("``\`\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            f.write("``\`\n\n")
        
        # Matrices de confusión
        f.write("## Matrices de Confusión\n\n")
        f.write("Ver imagen: \`all_confusion_matrices.png\`\n\n")
        
        # Curvas ROC y Precision-Recall
        f.write("## Curvas ROC y Precision-Recall\n\n")
        f.write("Ver imagen: \`model_comparison_dashboard.png\`\n\n")
        
        # Análisis de costo-beneficio
        f.write("## Análisis de Costo-Beneficio\n\n")
        f.write("Ver imagen: \`cost_benefit_analysis.png\`\n\n")
        
        # Conclusiones
        f.write("## Conclusiones\n\n")
        
        # Encontrar el mejor modelo según F1
        best_model = max(metrics_dict.items(), key=lambda x: x[1].get('f1', 0))
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]
        
        f.write(f"El mejor modelo según F1 Score es **{best_model_name}** con:\n\n")
        f.write(f"- F1 Score: {best_model_metrics.get('f1', 'N/A'):.4f}\n")
        f.write(f"- Precision: {best_model_metrics.get('precision', 'N/A'):.4f}\n")
        f.write(f"- Recall: {best_model_metrics.get('recall', 'N/A'):.4f}\n")
        if 'roc_auc' in best_model_metrics:
            f.write(f"- AUC-ROC: {best_model_metrics.get('roc_auc', 'N/A'):.4f}\n")
        
        # Encontrar el mejor modelo según AUC
        auc_models = {name: metrics.get('roc_auc', 0) for name, metrics in metrics_dict.items() if 'roc_auc' in metrics}
        if auc_models:
            best_auc_model_name = max(auc_models.items(), key=lambda x: x[1])[0]
            
            if best_auc_model_name != best_model_name:
                best_auc_model_metrics = metrics_dict[best_auc_model_name]
                f.write(f"\nEl mejor modelo según AUC-ROC es **{best_auc_model_name}** con:\n\n")
                f.write(f"- AUC-ROC: {best_auc_model_metrics.get('roc_auc', 'N/A'):.4f}\n")
                f.write(f"- F1 Score: {best_auc_model_metrics.get('f1', 'N/A'):.4f}\n")
                f.write(f"- Precision: {best_auc_model_metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"- Recall: {best_auc_model_metrics.get('recall', 'N/A'):.4f}\n")
        
        # Recomendaciones
        f.write("\n## Recomendaciones\n\n")
        f.write("1. **Selección de modelo**: Basado en los resultados, se recomienda utilizar el modelo ")
        if 'roc_auc' in best_model_metrics and best_model_metrics.get('roc_auc', 0) > 0.95:
            f.write(f"**{best_model_name}** para implementación en producción, ya que ofrece el mejor equilibrio entre precisión y recall.\n\n")
        else:
            f.write("con mejor F1 Score para casos donde se requiere equilibrio, o el modelo con mejor AUC para casos donde se necesita una buena discriminación general.\n\n")
        
        f.write("2. **Umbral de clasificación**: Ajustar el umbral de clasificación según el análisis de costo-beneficio para optimizar el rendimiento en el contexto específico de negocio.\n\n")
        
        f.write("3. **Monitoreo**: Implementar un sistema de monitoreo para detectar cambios en el rendimiento del modelo a lo largo del tiempo.\n\n")
        
        f.write("4. **Reentrenamiento**: Establecer un cronograma para reentrenar el modelo periódicamente con datos nuevos.\n\n")

def generate_html_report(metrics_dict: Dict[str, Dict[str, Any]], 
                        best_params: Dict[str, Dict[str, Any]],
                        output_path: str):
    """
    Genera un informe en formato HTML.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        best_params: Diccionario con mejores parámetros por modelo
        output_path: Ruta donde guardar el informe
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Evaluación de Modelos</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; }
            h3 { color: #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .code { background-color: #f8f8f8; padding: 15px; border-radius: 5px; font-family: monospace; overflow-x: auto; }
            .metric-good { color: #27ae60; font-weight: bold; }
            .metric-medium { color: #f39c12; font-weight: bold; }
            .metric-bad { color: #e74c3c; font-weight: bold; }
            .conclusion { background-color: #eaf2f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Informe de Evaluación de Modelos de Detección de Fraude</h1>
        
        <h2>Resumen de Métricas</h2>
        <table>
            <tr>
                <th>Modelo</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>F2</th>
                <th>AUC</th>
                <th>AP</th>
            </tr>
    """
    
    # Añadir filas de la tabla
    for name, metrics in metrics_dict.items():
        accuracy = metrics.get('accuracy', 'N/A')
        precision = metrics.get('precision', 'N/A')
        recall = metrics.get('recall', 'N/A')
        f1 = metrics.get('f1', 'N/A')
        f2 = metrics.get('f2', 'N/A')
        auc = metrics.get('roc_auc', 'N/A')
        ap = metrics.get('avg_precision', 'N/A')
        
        html += f"""
            <tr>
                <td>{name}</td>
        """
        
        # Añadir métricas con formato condicional
        for metric in [accuracy, precision, recall, f1, f2, auc, ap]:
            if isinstance(metric, (int, float)):
                if metric > 0.9:
                    html += f'<td class="metric-good">{metric:.4f}</td>'
                elif metric > 0.7:
                    html += f'<td class="metric-medium">{metric:.4f}</td>'
                else:
                    html += f'<td class="metric-bad">{metric:.4f}</td>'
            else:
                html += f'<td>{metric}</td>'
        
        html += """
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Mejores Hiperparámetros</h2>
    """
    
    # Añadir mejores parámetros
    for name, params in best_params.items():
        html += f"""
        <h3>{name}</h3>
        <div class="code">
        """
        
        for param, value in params.items():
            html += f"{param}: {value}<br>"
        
        html += """  value in params.items():
            html += f"{param}: {value}<br>"
        <h2>Visualizaciones</h2>
        
        <h3>Matrices de Confusión</h3>
        <img src="all_confusion_matrices.png" alt="Matrices de Confusión">
        
        <h3>Curvas ROC y Precision-Recall</h3>
        <img src="model_comparison_dashboard.png" alt="Dashboard de Comparación de Modelos">
        
        <h3>Análisis de Costo-Beneficio</h3>
        <img src="cost_benefit_analysis.png" alt="Análisis de Costo-Beneficio">
        
        <h2>Conclusiones</h2>
    """
    
    # Añadir conclusiones
    best_model = max(metrics_dict.items(), key=lambda x: x[1].get('f1', 0))
    best_model_name = best_model[0]
    best_model_metrics = best_model[1]
    
    html += f"""
        <div class="conclusion">
            <p>El mejor modelo según F1 Score es <strong>{best_model_name}</strong> con:</p>
            <ul>
                <li>F1 Score: {best_model_metrics.get('f1', 'N/A'):.4f}</li>
                <li>Precision: {best_model_metrics.get('precision', 'N/A'):.4f}</li>
                <li>Recall: {best_model_metrics.get('recall', 'N/A'):.4f}</li>
    """
    
    if 'roc_auc' in best_model_metrics:
        html += f"""
                <li>AUC-ROC: {best_model_metrics.get('roc_auc', 'N/A'):.4f}</li>
        """
    
    html += """
            </ul>
        </div>
    """
    
    # Añadir mejor modelo según AUC si es diferente
    auc_models = {name: metrics.get('roc_auc', 0) for name, metrics in metrics_dict.items() if 'roc_auc' in metrics}
    if auc_models:
        best_auc_model_name = max(auc_models.items(), key=lambda x: x[1])[0]
        
        if best_auc_model_name != best_model_name:
            best_auc_model_metrics = metrics_dict[best_auc_model_name]
            html += f"""
            <div class="conclusion">
                <p>El mejor modelo según AUC-ROC es <strong>{best_auc_model_name}</strong> con:</p>
                <ul>
                    <li>AUC-ROC: {best_auc_model_metrics.get('roc_auc', 'N/A'):.4f}</li>
                    <li>F1 Score: {best_auc_model_metrics.get('f1', 'N/A'):.4f}</li>
                    <li>Precision: {best_auc_model_metrics.get('precision', 'N/A'):.4f}</li>
                    <li>Recall: {best_auc_model_metrics.get('recall', 'N/A'):.4f}</li>
                </ul>
            </div>
            """
    
    # Añadir recomendaciones
    html += """
        <h2>Recomendaciones</h2>
        <ol>
            <li><strong>Selección de modelo</strong>: Basado en los resultados, se recomienda utilizar el modelo con mejor F1 Score para casos donde se requiere equilibrio, o el modelo con mejor AUC para casos donde se necesita una buena discriminación general.</li>
            <li><strong>Umbral de clasificación</strong>: Ajustar el umbral de clasificación según el análisis de costo-beneficio para optimizar el rendimiento en el contexto específico de negocio.</li>
            <li><strong>Monitoreo</strong>: Implementar un sistema de monitoreo para detectar cambios en el rendimiento del modelo a lo largo del tiempo.</li>
            <li><strong>Reentrenamiento</strong>: Establecer un cronograma para reentrenar el modelo periódicamente con datos nuevos.</li>
        </ol>
    </body>
    </html>
    """
    
    # Guardar archivo HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

# Ejemplo de uso
if __name__ == "__main__":
    from src.config import REPORTS_DIR
    import pickle
    
    # Cargar métricas y parámetros (asumiendo que se guardaron previamente)
    try:
        with open('models/best_metrics.pkl', 'rb') as f:
            best_metrics = pickle.load(f)
        
        with open('models/best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
        
        # Generar informes
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Informe Markdown
        md_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.md')
        generate_markdown_report(best_metrics, best_params, md_report_path)
        
        # Informe HTML
        html_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.html')
        generate_html_report(best_metrics, best_params, html_report_path)
        
        print(f"Informes generados en {REPORTS_DIR}")
    except Exception as e:
        print(f"Error al generar informes: {str(e)}")

EOF

# Crear script de entrenamiento de modelo base
cat <<EOF > src/model_training.py
"""
Funciones para el entrenamiento y optimización de modelos del proyecto $nombre.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional

EOF

# Crear script de utilidades base
cat <<EOF > src/utils.py
"""
Utilidades generales para el proyecto $nombre.
"""

import time
import os
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from src.config import FIGURES_DIR

def timer_decorator(func: Callable) -> Callable:
    """
    Decorador para medir el tiempo de ejecución de una función.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Función {func.__name__} ejecutada en {execution_time:.2f} segundos")
        return result
    return wrapper

def save_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
    """
    Guarda la figura actual de matplotlib en el directorio de figuras.
    
    Args:
        filename: Nombre del archivo
        dpi: Resolución de la imagen
        bbox_inches: Ajuste de bordes
        
    Returns:
        Ruta completa donde se guardó la figura
    """
    # Crear directorio si no existe
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Construir ruta completa
    filepath = os.path.join(FIGURES_DIR, filename)
    
    # Guardar figura
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figura guardada en {filepath}")
    
    return filepath

def detect_outliers_iqr(data: Union[pd.Series, np.ndarray], threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detecta outliers usando el método IQR.
    
    Args:
        data: Serie o array con datos
        threshold: Multiplicador para IQR
        
    Returns:
        Diccionario con información sobre outliers
    """
    # Convertir a numpy array si es necesario
    if isinstance(data, pd.Series):
        data_array = data.values
    else:
        data_array = data
    
    # Calcular cuartiles e IQR
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    # Calcular límites
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Identificar outliers
    outliers = np.logical_or(data_array < lower_bound, data_array > upper_bound)
    outliers_count = np.sum(outliers)
    outliers_percentage = outliers_count / len(data_array) * 100
    
    return {
        'count': outliers_count,
        'percentage': outliers_percentage,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mask': outliers
    }

def print_section_header(title: str, width: int = 80) -> None:
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title: Título de la sección
        width: Ancho del encabezado
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_step(step_number: int, step_description: str) -> None:
    """
    Imprime un paso numerado en un proceso.
    
    Args:
        step_number: Número del paso
        step_description: Descripción del paso
    """
    print(f"\n[Paso {step_number}] {step_description}")
    print("-" * 80)

def format_time(seconds: float) -> str:
    """
    Formatea un tiempo en segundos a un formato legible.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        Tiempo formateado
    """
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} minutos y {remaining_seconds:.2f} segundos"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)} horas, {int(minutes)} minutos y {seconds:.2f} segundos"

def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Calcula y formatea el uso de memoria de un DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Uso de memoria formateado
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    # Convertir a unidades legibles
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024**2:
        return f"{memory_bytes/1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes/1024**2:.2f} MB"
    else:
        return f"{memory_bytes/1024**3:.2f} GB"
EOF

# Crear script de visualización base
cat <<EOF > src/visualization_helpers.py
"""
Funciones auxiliares para visualización de resultados de modelos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.utils import save_figure

def plot_all_confusion_matrices(metrics_dict: Dict[str, Dict[str, Any]], figsize=(15, 10)):
    """
    Visualiza matrices de confusión para múltiples modelos en una sola figura.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        figsize: Tamaño de la figura
    """
    # Determinar número de modelos y configurar subplots
    n_models = len(metrics_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        if i < len(axes):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
            axes[i].set_title(f'Matriz de Confusión - {name}')
            axes[i].set_ylabel('Valor Real')
            axes[i].set_xlabel('Valor Predicho')
    
    # Ocultar ejes no utilizados
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    save_figure('all_confusion_matrices.png')
    return fig

def plot_all_roc_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(10, 8)):
    """
    Visualiza curvas ROC para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Graficar línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio')
    
    for name, metrics in metrics_dict.items():
        if metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                print(f"Advertencia: No se encontraron etiquetas verdaderas para el modelo {name}. Saltando.")
                continue
                
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, metrics['y_prob'])
            
            # Calcular AUC si no está en las métricas
            if 'roc_auc' in metrics:
                auc = metrics['roc_auc']
            else:
                auc = roc_auc_score(y_true, metrics['y_prob'])
            
            # Graficar curva ROC
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Múltiples Modelos')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_roc_curves.png')
    
    return plt.gcf()

def plot_all_precision_recall_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(10, 8)):
    """
    Visualiza curvas de precisión-recall para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    for name, metrics in metrics_dict.items():
        if metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                print(f"Advertencia: No se encontraron etiquetas verdaderas para el modelo {name}. Saltando.")
                continue
                
            # Calcular curva precision-recall
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_prob'])
            
            # Calcular AP si no está en las métricas
            if 'avg_precision' in metrics:
                ap = metrics['avg_precision']
            else:
                ap = average_precision_score(y_true, metrics['y_prob'])
            
            # Graficar curva precision-recall
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap:.4f})')
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall para Múltiples Modelos')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_precision_recall_curves.png')
    
    return plt.gcf()

def create_model_comparison_dashboard(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(15, 12)):
    """
    Crea un dashboard completo con comparación de modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    fig = plt.figure(figsize=figsize)
    
    # Definir grid para subplots
    gs = fig.add_gridspec(3, 2)
    
    # 1. Comparación de métricas principales
    ax1 = fig.add_subplot(gs[0, :])
    
    # Preparar datos para gráfico de barras
    models = list(metrics_dict.keys())
    metrics_to_plot = ['precision', 'recall', 'f1', 'roc_auc']
    metrics_data = {metric: [] for metric in metrics_to_plot}
    
    for model in models:
        for metric in metrics_to_plot:
            if metric in metrics_dict[model]:
                metrics_data[metric].append(metrics_dict[model][metric])
            else:
                metrics_data[metric].append(0)
    
    # Crear gráfico de barras agrupadas
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i*width - width*1.5, metrics_data[metric], width, label=metric.upper())
    
    ax1.set_title('Comparación de Métricas Principales')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Curvas ROC
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    
    for name, metrics in metrics_dict.items():
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                continue
                
            fpr, tpr, _ = roc_curve(y_true, metrics['y_prob'])
            auc = metrics.get('roc_auc', roc_auc_score(y_true, metrics['y_prob']))
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    ax2.set_title('Curvas ROC para Múltiples Modelos')
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Curvas Precision-Recall
    ax3 = fig.add_subplot(gs[1, 1])
    
    for name, metrics in metrics_dict.items():
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                continue
                
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_prob'])
            ap = metrics.get('avg_precision', average_precision_score(y_true, metrics['y_prob']))
            ax3.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
    
    ax3.set_title('Curvas Precision-Recall para Múltiples Modelos')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Tabla de métricas detalladas
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Preparar datos para tabla
    table_data = []
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc', 'avg_precision']
    
    for model in models:
        row = [model]
        for metric in metrics_list:
            if metric in metrics_dict[model]:
                row.append(f"{metrics_dict[model][metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'AUC', 'AP'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    
    # Guardar figura
    save_figure('model_comparison_dashboard.png')
    
    return fig

def plot_model_comparison_heatmap(metrics_dict: Dict[str, Dict[str, Any]], figsize=(12, 8)):
    """
    Crea un mapa de calor para comparar métricas entre modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        figsize: Tamaño de la figura
    """
    # Preparar datos para el mapa de calor
    metrics_to_include = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'mcc', 'kappa', 'roc_auc', 'avg_precision']
    models = list(metrics_dict.keys())
    
    # Crear DataFrame
    data = []
    for model in models:
        row = {}
        for metric in metrics_to_include:
            if metric in metrics_dict[model]:
                row[metric] = metrics_dict[model][metric]
            else:
                row[metric] = np.nan
        data.append(row)
    
    df = pd.DataFrame(data, index=models)
    
    # Crear mapa de calor
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.4f', cmap='Blues', linewidths=0.5, cbar=True)
    plt.title('Comparación Detallada de Modelos')
    plt.tight_layout()
    
    # Guardar figura
    save_figure('model_comparison_heatmap.png')
    
    return plt.gcf()

def plot_cost_benefit_analysis(metrics_dict: Dict[str, Dict[str, Any]], 
                              y_test: np.ndarray = None,
                              cost_fp: float = 10,
                              cost_fn: float = 100,
                              benefit_tp: float = 50,
                              benefit_tn: float = 1,
                              model_name: str = None,
                              figsize=(10, 6)):
    """
    Realiza un análisis de costo-beneficio para diferentes umbrales de clasificación.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        cost_fp: Costo de un falso positivo
        cost_fn: Costo de un falso negativo
        benefit_tp: Beneficio de un verdadero positivo
        benefit_tn: Beneficio de un verdadero negativo
        model_name: Nombre del modelo a analizar (si es None, se usa el mejor según AUC)
        figsize: Tamaño de la figura
    """
    # Si no se especifica un modelo, usar el mejor según AUC
    if model_name is None:
        best_model = max(metrics_dict.items(), key=lambda x: x[1].get('roc_auc', 0))
        model_name = best_model[0]
    
    # Obtener probabilidades y etiquetas verdaderas
    metrics = metrics_dict[model_name]
    if metrics['y_prob'] is None:
        print(f"El modelo {model_name} no tiene probabilidades predichas. No se puede realizar el análisis.")
        return None
    
    # Determinar las etiquetas verdaderas
    if 'y_test' in metrics:
        y_true = metrics['y_test']
    elif 'y_true' in metrics:
        y_true = metrics['y_true']
    elif y_test is not None:
        y_true = y_test
    else:
        print(f"No se encontraron etiquetas verdaderas para el modelo {model_name}.")
        return None
    
    # Calcular costo-beneficio para diferentes umbrales
    thresholds = np.linspace(0.01, 0.99, 20)
    total_costs = []
    
    for threshold in thresholds:
        y_pred = (metrics['y_prob'] >= threshold).astype(int)
        
        # Calcular matriz de confusión
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calcular costo total
        total_cost = (cost_fp * fp) + (cost_fn * fn) - (benefit_tp * tp) - (benefit_tn * tn)
        total_costs.append(total_cost)
    
    # Encontrar umbral óptimo
    optimal_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = total_costs[optimal_idx]
    
    # Crear gráfico
    plt.figure(figsize=figsize)
    plt.plot(thresholds, total_costs, 'r-o')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--')
    plt.axhline(y=0, color='k', linestyle=':')
    plt.title('Análisis de Costo-Beneficio')
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Costo Total')
    plt.grid(True, alpha=0.3)
    
    # Añadir anotación para umbral óptimo
    plt.annotate(f'Umbral óptimo: {optimal_threshold:.2f}\nCosto: {optimal_cost:.2f}',
                xy=(optimal_threshold, optimal_cost),
                xytext=(optimal_threshold + 0.1, optimal_cost + 100),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    # Guardar figura
    save_figure('cost_benefit_analysis.png')
    
    return plt.gcf(), optimal_threshold

def plot_calibration_curves(metrics_dict, y_test=None, n_bins=10, figsize=(12, 8)):
    """
    Visualiza las curvas de calibración para todos los modelos.
    
    Args:
        metrics_dict (dict): Diccionario con métricas de todos los modelos.
        y_test (array): Etiquetas reales para el conjunto de prueba.
        n_bins (int): Número de bins para la calibración.
        figsize (tuple): Tamaño de la figura.
        
    Returns:
        matplotlib.figure.Figure: Figura con las curvas de calibración.
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    # Colores para las líneas
    colors = plt.cm.get_cmap('tab10', len(metrics_dict))
    
    # Línea de referencia (calibración perfecta)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Graficar curva de calibración para cada modelo
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            y_prob = metrics['y_prob']
            
            # Si no se proporciona y_test, intentar obtenerlo de las métricas
            if y_test is None and 'y_true' in metrics:
                y_true = metrics['y_true']
            else:
                y_true = y_test
                
            if y_true is not None:
                # Calcular curva de calibración
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
                
                # Graficar curva
                plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                         color=colors(i), label=f"{name}")
    
    # Configurar gráfico
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    return plt.gcf()
EOF

# Detectar si estás en WSL
if grep -qi microsoft /proc/version; then
  echo ""
  echo "🧠 Estás en WSL. Asegúrate de tener instaladas estas extensiones en VS Code dentro de WSL:"
  echo "- Python (Microsoft)"
  echo "- Jupyter (Microsoft)"
  echo "- Pylance"
  echo ""
fi

# Inicializar repositorio Git
git init
git add .
git commit -m "Primer commit: estructura inicial del proyecto de ciencia de datos"

echo "✅ Proyecto '$nombre' creado correctamente con entorno virtual, VS Code configurado y Git inicializado."

