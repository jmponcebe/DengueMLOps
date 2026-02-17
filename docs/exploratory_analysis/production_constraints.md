# Restricciones Temporales para Producción

## El Problema del "Future Information Leakage"

En el análisis retrospectivo, es fácil **sobreestimar** la capacidad predictiva porque tenemos acceso a información que **NO estaría disponible** en el momento real de hacer una predicción.

### Escenario Real de Predicción

**Situación**: Es lunes 15 de abril de 2024, queremos predecir el nivel de alerta para la semana del 22-28 de abril.

**❌ NO tenemos**:

- Casos de dengue de la semana objetivo (22-28 abril)
- Índices epidemiológicos actuales (receptividad, transmisión)
- Clima de la semana objetivo
- Alertas sanitarias actuales

**✅ SÍ tenemos**:

- Historial climático hasta 4-8 semanas atrás (marzo)
- Datos poblacionales constantes
- Información temporal (es abril, semana 16)
- Patrones históricos hasta la fecha actual

### Implicaciones Epidemiológicas

**Lag de incubación vectorial**:

- Huevos → larvas → pupas → mosquito adulto: 8-12 días
- Período de incubación extrínseca: 8-12 días  
- **Total**: 4-6 semanas desde condiciones climáticas hasta transmisión

**Lag de detección y reporte**:

- Incubación en humanos: 4-7 días
- Síntomas → diagnóstico → reporte: 1-2 semanas
- **Total adicional**: 2-3 semanas

### Estrategia de Variables Temporalmente Válidas

```python
# Ejemplo de variables válidas para predicción de alertas semana N
variables_disponibles = {
    'demograficas_constantes': ['poblacion', 'uf', 'municipio'],
    'temporales_conocidas': ['mes', 'semana_epidemiologica', 'año'],
    'climaticas_lag': [
        'temp_lag_4w', 'temp_lag_6w', 'temp_lag_8w',
        'humid_lag_4w', 'humid_lag_6w', 'humid_lag_8w'
    ],
    'historicas_agregadas': [
        'casos_historicos_mismo_mes', 
        'patron_estacional_historico',
        'intensidad_año_anterior'
    ]
}

# Variables que crearían DATA LEAKAGE
variables_prohibidas = {
    'epidemiologicas_actuales': ['nivel_alerta', 'casos_dengue', 'transmision'],
    'climaticas_simultaneas': ['temp_actual', 'humid_actual'],
    'agregaciones_futuras': ['casos_mes_completo', 'pico_estacional_año']
}
```

### Impacto en Performance Esperada

- **Correlaciones realistas**: 0.05-0.31 (vs 0.25-0.51 en análisis retrospectivo)
- **Accuracy esperada**: 45-65% (vs >80% con variables prohibidas)
- **Valor real**: Sistema de alerta temprana epidemiológicamente válido

### Validación de Enfoque

Este enfoque es **científicamente correcto** porque:

1. Respeta los principios de causalidad temporal epidemiológica
2. Evita optimismo artificial en métricas de evaluación  
3. Genera un sistema realmente deployable en producción
4. Mantiene coherencia con literatura científica sobre predicción de enfermedades vectoriales
