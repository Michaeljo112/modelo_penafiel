# Modelo Penafiel

Codigo del modelo basado en agentes sobre heterogeneidad moral, instituciones endogenas y sobreproduccion de elites.

## Archivos

- `simulation.py`: modelo central y dinamica de simulacion.
- `run_mc.py`: ejecucion Monte Carlo de escenarios principales y ablaciones.
- `sensitivity.py`: barridos de sensibilidad de parametros.
- `make_figures.py`: generacion de figuras desde resultados simulados.
- `requirements.txt`: dependencias de Python.

## Reproducibilidad

```bash
python run_mc.py
python run_mc.py --original
python run_mc.py --no-hierarchy
python sensitivity.py
python make_figures.py
```

Los scripts generan salidas locales en `results/` y `figures/`, que no se versionan en este repositorio.

Los resultados del articulo fueron generados con Python 3.10.8, NumPy 2.1.3, pandas 2.2.3 y matplotlib 3.10.0.
