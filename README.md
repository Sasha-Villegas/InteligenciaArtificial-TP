# TP3 – Modelo de Hopfield (Reconocimiento de Patrones)

**Autor:** Sasha Villegas Basso  
**Materia:** Inteligencia Artificial  
**Universidad:** Nacional de Río Cuarto  

## 🧠 Descripción
Prototipo implementado desde cero del **modelo de red de Hopfield**, orientado a la recuperación de imágenes binarias simples.  
El modelo demuestra cómo una red puede memorizar y recuperar patrones a partir de versiones ruidosas.

## ⚙️ Requisitos
```bash
pip install numpy



# TP2 – Búsqueda en el espacio de estados (línea de montaje 1D)

Este repositorio implementa y compara:
- Búsqueda exhaustiva (zig-zag simétrica) sobre el eje H.
- Búsqueda heurística (Greedy y A* 1D) con una estimación de Δx simulada (ruido opcional).

## Requisitos
- Python 3.9+ (sin dependencias externas)

## Uso rápido
Comparación completa:
    python tp2_search.py --compare

Parámetros clave:
- --A (objetivo), --B (inicio), --epsilon (tolerancia), --step (tamaño de paso),
- --noise (ruido de la heurística), --seed (reproducibilidad).

Ejemplos:
    python tp2_search.py --A 3.2 --B 0.0 --epsilon 0.2 --step 0.5 --compare
    python tp2_search.py --greedy --step 0.5 --noise 0.1
    python tp2_search.py --exhaustive

## Diseño
- Entorno 1D con verificación |x - A| <= epsilon.
- Exhaustiva: B, B+ΔH, B-ΔH, B+2ΔH, ...
- Heurística:
  - Greedy: x ← x + clip(gain * estimate, ±max_step)
  - A*: discretización 1D con h(x)=|A-x| (admisible sin ruido).

## Notas
- El ruido (--noise) simula incertidumbre sensorial (visión/relieve).
- En líneas reales, Greedy/A* reducen iteraciones; la exhaustiva es un fallback seguro.
