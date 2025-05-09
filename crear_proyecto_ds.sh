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

# Crear script de configuracion base
cat <<EOF > src/config.py
import pandas as pd

EOF

# Crear script de preparación de datos base
cat <<EOF > src/data_prep.py
import pandas as pd

EOF

# Crear script de evaluación de datos base
cat <<EOF > src/evaluate.py
import pandas as pd

EOF

# Crear script de generación de informes base
cat <<EOF > src/generate_report.py
import pandas as pd

EOF

# Crear script de entrenamiento de modelo base
cat <<EOF > src/model_training.py
import pandas as pd

EOF

# Crear script de utilidades base
cat <<EOF > src/utils.py
import pandas as pd

EOF

# Crear script de visualización base
cat <<EOF > src/visualization_helpers.py
import pandas as pd

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

