"""
API client for mosqlimate - Fetches dengue data directly from the API
Used for initial data download and periodic updates
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar logger para que funcione en notebooks
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MosqlimateAPIClient:
    """
    Client for fetching data from mosqlimate API
    API Docs: https://api.mosqlimate.org/docs
    """

    BASE_URL = "https://api.mosqlimate.org/api"

    # 27 estados brasileños
    STATES = [
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
        'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
        'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client

        Args:
            api_key: X-UID-Key from https://api.mosqlimate.org/<username>/
                     Si no se provee, se carga desde variable de entorno MOSQLIMATE_API_KEY
        """
        # Usar api_key provisto o cargar desde variable de entorno
        self.api_key = api_key or os.getenv('MOSQLIMATE_API_KEY')
        self.session = requests.Session()

        if self.api_key:
            # Header correcto según documentación: X-UID-Key
            self.session.headers.update({
                'X-UID-Key': self.api_key
            })

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying... ({e})")
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_available_datasets(self) -> List[Dict]:
        """
        Get list of available datasets

        Returns:
            List of dataset metadata
        """
        try:
            data = self._make_request("datastore/")
            if data is None:
                return []
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error fetching datasets: {e}")
            return []

    def fetch_dengue_data(
        self,
        geocode: Optional[int] = None,
        uf: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        disease: str = "dengue",
        per_page: int = 100
    ) -> pd.DataFrame:
        """
        Fetch dengue data from API with automatic pagination

        Args:
            geocode: Municipality geocode (IBGE code)
            uf: State code (e.g., 'SP', 'RJ', 'AC')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            disease: Disease name (default: 'dengue')
            per_page: Items per page (max 100)

        Returns:
            DataFrame with dengue data
        """
        # Parámetros base requeridos por la API
        params = {
            'disease': disease,
            'page': 1,
            'per_page': per_page
        }

        # Parámetros opcionales
        if geocode:
            params['geocode'] = geocode
        if uf:
            params['uf'] = uf
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        all_data = []

        try:
            # Primera petición para obtener paginación
            data = self._make_request("datastore/infodengue/", params=params)

            if data is not None and 'items' in data:
                all_data.extend(data['items'])

                # Obtener info de paginación
                pagination = data.get('pagination', {})
                total_pages = pagination.get('total_pages', 1)

                # Si hay más páginas, obtenerlas
                for page in range(2, total_pages + 1):
                    params['page'] = page
                    data = self._make_request("datastore/infodengue/", params=params)
                    if data is not None and 'items' in data:
                        all_data.extend(data['items'])

                logger.info(f"Fetched {len(all_data)} records for uf={uf} ({total_pages} pages)")

            return pd.DataFrame(all_data)

        except Exception as e:
            logger.error(f"Error fetching dengue data: {e}")
            return pd.DataFrame()

    def _get_missing_months(
        self,
        output_path: Path,
        states: List[str],
        year_start: int,
        year_end: int
    ) -> List[tuple]:
        """
        Identifica qué meses faltan descargar

        Returns:
            Lista de tuplas (state, year, month) que no existen localmente
        """
        missing = []
        now = datetime.now()

        # Último mes completo disponible (mes anterior al actual)
        last_available_month = (now.replace(day=1) - timedelta(days=1))

        for state in states:
            for year in range(year_start, year_end + 1):
                for month in range(1, 13):
                    check_date = datetime(year, month, 1)

                    # Solo buscar datos hasta el mes pasado (excluir mes actual y futuros)
                    if check_date > last_available_month:
                        continue

                    # Verificar si existe el archivo
                    file_path = output_path / f"uf={state}" / f"year={year}" / f"month={month:02d}" / "data.parquet"
                    if not file_path.exists():
                        missing.append((state, year, month))

        return missing

    def sync_data(
        self,
        states: Optional[List[str]] = None,
        year_start: int = 2010,
        output_path: Optional[Union[str, Path]] = None,
        save_format: str = "parquet"
    ):
        """
        Sincroniza datos: descarga completo si no hay nada, actualiza solo lo faltante si hay datos
        Función inteligente que detecta automáticamente qué necesita descargar

        Args:
            states: List of state codes. If None, uses all
            year_start: Start year
            output_path: Base path to save data
            save_format: Format to save ('parquet' recommended)
        """
        if states is None:
            states = self.STATES

        if output_path is None:
            output_path = "../data/raw/historical_api_data"

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Año final = año actual
        year_end = datetime.now().year

        # Detectar qué meses faltan
        logger.info("🔍 Detectando datos faltantes...")
        missing = self._get_missing_months(output_path, states, year_start, year_end)

        if not missing:
            logger.info("✅ Todos los datos están actualizados")
            return

        # Mostrar detalles de archivos faltantes
        total_existing = (year_end - year_start + 1) * 12 * len(states) - len(missing)
        logger.info(f"📊 Estado: {total_existing} archivos existentes, {len(missing)} faltantes")

        # Agrupar por año/mes para mostrar
        missing_by_period = {}
        for state, year, month in missing:
            key = f"{year}-{month:02d}"
            if key not in missing_by_period:
                missing_by_period[key] = []
            missing_by_period[key].append(state)

        logger.info("📅 Períodos faltantes:")
        for period in sorted(missing_by_period.keys())[:5]:  # Mostrar primeros 5
            states_list = missing_by_period[period]
            logger.info(f"   {period}: {len(states_list)} estados ({', '.join(states_list[:5])}{'...' if len(states_list) > 5 else ''})")
        if len(missing_by_period) > 5:
            logger.info(f"   ... y {len(missing_by_period) - 5} períodos más")

        logger.info(f"⬇️  Descargando {len(missing)} archivos...")

        # Descargar solo lo que falta
        for state, year, month in tqdm(missing, desc="Descargando"):
            date_start = f"{year}-{month:02d}-01"

            # Último día del mes
            if month == 12:
                date_end = f"{year}-12-31"
            else:
                next_month = datetime(year, month, 1) + timedelta(days=32)
                date_end = (next_month.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")

            # Fetch data
            df = self.fetch_dengue_data(
                uf=state,
                start=date_start,
                end=date_end
            )

            if not df.empty:
                # Crear estructura de directorios
                save_dir = output_path / f"uf={state}" / f"year={year}" / f"month={month:02d}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Guardar archivo
                if save_format == "parquet":
                    save_path = save_dir / "data.parquet"
                    df.to_parquet(save_path, index=False)
                else:
                    save_path = save_dir / "data.csv"
                    df.to_csv(save_path, index=False)

                logger.debug(f"Guardado {save_path}")

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"\n✅ Sincronización completa. Datos guardados en: {output_path}")
        logger.info(f"📦 Total archivos descargados: {len(missing)}")


def sync_dataset(year_start: int = 2010, output_path: Optional[str] = None):
    """
    Sincroniza dataset de forma inteligente
    - Si no hay datos: descarga todo desde year_start
    - Si hay datos: solo descarga meses faltantes

    Requiere MOSQLIMATE_API_KEY en .env o variable de entorno

    Args:
        year_start: Año inicial (default 2010)
        output_path: Ruta donde guardar datos
    """
    # Cliente carga automáticamente API key desde .env
    client = MosqlimateAPIClient()

    if not client.api_key:
        print("❌ ERROR: API key no encontrada")
        print()
        print("Configura MOSQLIMATE_API_KEY en archivo .env:")
        print("  MOSQLIMATE_API_KEY=tu_uid_key_aqui")
        print()
        print("O como variable de entorno:")
        print("  export MOSQLIMATE_API_KEY='tu_uid_key_aqui'")
        return

    client.sync_data(year_start=year_start, output_path=output_path)


if __name__ == "__main__":
    # API key desde variable de entorno o hardcoded para test
    api_key = os.environ.get('MOSQLIMATE_API_KEY')

    if not api_key:
        print("⚠️  Para usar la API necesitas un X-UID-Key")
        print()
        print("Cómo obtenerlo:")
        print("1. Ir a https://api.mosqlimate.org/")
        print("2. Autenticarse con GitHub")
        print("3. Ir a tu perfil: https://api.mosqlimate.org/<tu_usuario>/")
        print("4. Copiar el UID-Key")
        print()
        print("Usar: export MOSQLIMATE_API_KEY='tu_key_aqui'")
        print("  o:  MosqlimateAPIClient(api_key='tu_key_aqui')")
    else:
        print("🧪 Test del cliente API...")

        client = MosqlimateAPIClient(api_key=api_key)
        df = client.fetch_dengue_data(
            uf='SP',
            start='2024-01-01',
            end='2024-01-31',
            disease='dengue'
        )

        if not df.empty:
            print(f"✅ Descargados {len(df)} registros para SP (enero 2024)")
            print(f"Columnas: {list(df.columns)[:5]}...")
        else:
            print("❌ No se obtuvieron datos - verificar API key")
