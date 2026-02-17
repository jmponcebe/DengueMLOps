"""
Configuración global de pytest.
Asegura que el directorio raíz del proyecto esté en sys.path
para poder importar 'src' como paquete.
"""

import sys
from pathlib import Path

# Añade raíz del proyecto al path para imports tipo 'from src.X import ...'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
