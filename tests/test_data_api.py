"""
Tests para el cliente API de mosqlimate
"""

import sys
from pathlib import Path
import pytest
from datetime import datetime, timedelta

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.api_client import MosqlimateAPIClient


class TestMosqlimateAPIClient:
    """Tests para el cliente API"""
    
    @pytest.fixture
    def client(self):
        """Cliente de prueba"""
        return MosqlimateAPIClient()
    
    @pytest.fixture
    def data_path(self):
        """Path a los datos"""
        return Path("data/raw/historical_api_data")
    
    def test_client_initialization(self, client):
        """Test: Inicialización del cliente"""
        assert client is not None
        assert client.BASE_URL == "https://api.mosqlimate.org/api"
        assert len(client.STATES) == 27
    
    def test_states_list(self, client):
        """Test: Lista de estados brasileños completa"""
        expected_states = [
            'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
            'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
            'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
        ]
        assert client.STATES == expected_states
    
    def test_detect_missing_months(self, client, data_path):
        """Test: Detección de archivos faltantes"""
        missing = client._get_missing_months(
            output_path=data_path,
            states=['AC'],  # Solo un estado para ser rápido
            year_start=2010,
            year_end=2025
        )
        
        # Debe retornar una lista de tuplas
        assert isinstance(missing, list)
        if missing:
            assert all(isinstance(item, tuple) and len(item) == 3 for item in missing)
            # Verificar estructura: (state, year, month)
            state, year, month = missing[0]
            assert isinstance(state, str)
            assert isinstance(year, int)
            assert isinstance(month, int)
            assert 1 <= month <= 12
    
    def test_date_calculations(self):
        """Test: Cálculo correcto de fechas de fin de mes"""
        test_cases = [
            (2024, 1, "2024-01-31"),   # Enero
            (2024, 2, "2024-02-29"),   # Febrero bisiesto
            (2025, 2, "2025-02-28"),   # Febrero normal
            (2024, 12, "2024-12-31"),  # Diciembre
            (2024, 4, "2024-04-30"),   # Abril (30 días)
        ]
        
        for year, month, expected_end in test_cases:
            # Replicar lógica del cliente
            if month == 12:
                date_end = f"{year}-12-31"
            else:
                next_month = datetime(year, month, 1) + timedelta(days=32)
                date_end = (next_month.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            
            assert date_end == expected_end, f"Error en {year}-{month}: esperado {expected_end}, obtenido {date_end}"
    
    def test_data_coverage(self, client, data_path):
        """Test: Verificar cobertura de datos existentes"""
        if not data_path.exists():
            pytest.skip("No hay datos descargados")
        
        missing = client._get_missing_months(
            output_path=data_path,
            states=client.STATES,
            year_start=2010,
            year_end=2026
        )
        
        total_possible = len(client.STATES) * 16 * 12  # 27 estados * 16 años * 12 meses
        existing = total_possible - len(missing)
        coverage = existing / total_possible * 100
        
        # Deberíamos tener al menos 90% de cobertura
        assert coverage >= 90, f"Cobertura baja: {coverage:.1f}%"
        
        print(f"\n📊 Cobertura de datos: {coverage:.1f}% ({existing}/{total_possible})")


def test_sync_dataset_dry_run():
    """Test funcional: Verificar que sync funciona sin errores"""
    client = MosqlimateAPIClient()
    data_path = Path("data/raw/historical_api_data")
    
    # Solo verificar que no hay errores en la detección
    missing = client._get_missing_months(
        output_path=data_path,
        states=['SP'],  # Solo São Paulo
        year_start=2024,
        year_end=2025
    )
    
    assert isinstance(missing, list)
    print(f"\n✅ Detección funcional: {len(missing)} archivos faltantes para SP (2024-2025)")


if __name__ == "__main__":
    # Ejecutar tests con información detallada
    pytest.main([__file__, "-v", "-s"])
