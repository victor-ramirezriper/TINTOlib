
## Prerrequisitos

Verificar e instalar Python y Git

python --version
git --version

Si no tienes Python 3.12.10 (o una versión compatible de Python 3), descárgalo desde la web oficial de Python. Importante: Durante la instalación, marca la casilla inferior que dice "Add python.exe to PATH".

Si no tienes Git, descárgalo e instálalo desde Git for Windows.

Crear y activar el entorno virtual en PowerShell

cd D:\EXP_TINTO
python -m venv tinto_env

tinto_env\Scripts\Activate.ps1

Nota sobre permisos en PowerShell: Si te aparece un error indicando que la ejecución de scripts está deshabilitada, ejecuta primero este comando una sola vez: Set-ExecutionPolicy Unrestricted -Scope CurrentUser, y vuelve a intentar activar el entorno.

Crear el archivo requirements.txt

pandas>=2.0.0
seaborn>=0.12.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
mpi4py>=4.0.0
impi-rt>=2021.10.0
TINTOlib[refined]

Instalar las librerías con pip

python -m pip install --upgrade pip
pip install -r requirements.txt

Verificación de mpi4py en Windows
Para comprobar que mpi4py y el motor de ejecución MPI se enlazaron correctamente en tu sistema Windows, ejecuta este comando rápido en PowerShell:

python -c "from mpi4py import MPI; print('MPI configurado con éxito. Rango:', MPI.COMM_WORLD.Get_rank())"

Si el comando se ejecuta y devuelve el número de proceso (por ejemplo, Rango: 0) sin arrojar errores de librerías DLL faltantes, el entorno está configurado a la perfección.


