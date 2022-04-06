# Dactilología en lengua de señas mexicana usando visión por computadora  y máquinas de soporte vectorial

## Instalación
Se sugiere ejecutar el código en una máquina Linux con anaconda instalado, para instalar las librerías es necesario ejecutar los siguientes comandos en el orden en que se muestran:

```
conda install -c anaconda scikit-learn
conda install -c anaconda joblib
conda install -c anaconda ipykernel 
conda install -c conda-forge notebook
pip install pygame
pip install gTTS
conda install -c conda-forge pandas 
pip install mediapipe
pip install opencv-python
conda install -c conda-forge pynput
```

Posteriormente es necesario generar los modelos de clasificación ejecutando el siguiente comando:
```
python create_models.py
```

Una vez creados los modelos es posible crear el escript ejecutando el siguiente comando:
```
hand_gesture_recognition.py
```
