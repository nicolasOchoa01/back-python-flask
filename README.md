borrar carpeta venv
--> volver a generar enviroment python -m venv venv
--> instalar Libs de cada version contenida en este archivo, ejemplo: python -m pip install 'blinker==1.8.2'

comands

activa enviroment
.\venv\Scripts\activate

desactivar enviroment
deactivate

lista libs instaladas en enviroment
pip freeze

genera requirements.txt con detalle de cada lib instalada
pip freeze > requirements.txt

instalar libs contenidas dentro de requierements.txt
pip install -r requirements.txt

MORE INFORMATION -> https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

