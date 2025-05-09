# 🧰 Utilidades para Proyectos de Ciencia de Datos

Este repositorio contiene herramientas, scripts y plantillas que utilizo para iniciar y mantener mis proyectos de ciencia de datos de forma ordenada y eficiente.

Mi objetivo es construir una colección de utilidades reutilizables que me ayuden a acelerar el desarrollo de nuevos proyectos, aplicar buenas prácticas y mantener consistencia en mi portfolio profesional.

---

## 🚀 Utilidades disponibles

### 1. `crear_proyecto_ds.sh`

Script en Bash para crear automáticamente la estructura base de un proyecto de ciencia de datos.

🔧 Funcionalidades:
- Crea estructura de carpetas estándar (`data/`, `notebooks/`, `src/`, etc.)

- Configura entorno virtual con `venv`

- Instala librerías clave como `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, y `jupyterlab`

- Añade archivo `.gitignore` y configuración de VS Code

- Crea los notebooks base que tiene un proyecto de ciencia de datos, así como sus archivos de configuración en src/

- Inicializa un repositorio Git con primer commit

- Detecta si estás en **WSL** y recuerda instalar las extensiones necesarias

📥 Uso:
```bash
chmod +x crear_proyecto_ds.sh
./crear_proyecto_ds.sh
```

##✍️ Futuras utilidades

- Plantillas para notebooks de EDA

- Scripts de validación y limpieza avanzada

- Automatización de experimentos de ML

- Integración con GitHub o DVC

- Scripts de despliegue o testing


##🧠 Autor

Daniel Sánchez McCambridge

Apasionado de la inteligencia artificial, el análisis de datos y la automatización.\n
Actualmente desarrollando proyectos para portfolio y preparándome para entrevistas técnicas.


##📄 Licencia

Este repositorio está disponible bajo la licencia MIT. Eres libre de usar, modificar y compartir cualquier utilidad, con el debido reconocimiento.
