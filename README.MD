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

### 2. 🖥️ setup_wsl_terminal.sh – Mejora visual y funcional de la terminal en WSL

Este script automatiza la configuración de un entorno de terminal moderno y funcional para WSL (Windows Subsystem for Linux), ideal para desarrolladores y científicos de datos que trabajan en entornos mixtos Windows/Linux.

🔧 Funciones principales:

- Instala el shell zsh y el gestor de configuración Oh My Zsh

- Configura el tema visual Powerlevel10k, compatible con Git

- Añade plugins de productividad:

- zsh-autosuggestions (sugerencias de comandos)

- zsh-syntax-highlighting (resaltado de sintaxis)

- Actualiza automáticamente el archivo ~/.zshrc

- Cambia tu shell predeterminado a zsh

- Incluye mensajes informativos paso a paso

✅ Resultado: una terminal bonita, informativa y productiva, con indicadores de estado Git directamente en el prompt, sugerencias inteligentes, y soporte para alias personalizados.

📌 Requiere instalar la fuente MesloLGS NF en tu terminal para un renderizado óptimo del tema Powerlevel10k.

## ✍️ Futuras utilidades

- Plantillas para notebooks de EDA

- Scripts de validación y limpieza avanzada

- Automatización de experimentos de ML

- Integración con GitHub o DVC

- Scripts de despliegue o testing


## 🧠 Autor

Daniel Sánchez McCambridge

Apasionado de la inteligencia artificial, el análisis de datos y la automatización.\n
Actualmente desarrollando proyectos para portfolio y preparándome para entrevistas técnicas.


## 📄 Licencia

Este repositorio está disponible bajo la licencia MIT. Eres libre de usar, modificar y compartir cualquier utilidad, con el debido reconocimiento.
