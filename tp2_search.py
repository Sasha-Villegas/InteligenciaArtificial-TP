#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP2 – Búsqueda exhaustiva vs heurística (simulación 1D)
Autor: Sasha Villegas Basso
Descripción:
- Entorno 1D con objetivo A, posición inicial B sobre eje H.
- Verificación de encastre: |x - A| <= epsilon.
- Dos métodos:
    1) Exhaustiva (zig-zag simétrica) con paso ΔH.
    2) Heurística (Greedy o A* 1D) usando una estimación con ruido.
- CLI para correr y comparar.
Sin dependencias externas (stdlib).
"""

from dataclasses import dataclass
import random
import math
import argparse
from typing import List, Tuple, Optional


# -----------------------------
#  Entorno de simulación 1D
# -----------------------------
@dataclass
class LineAssemblyEnv:
    A: float                 # Posición verdadera del punto de montaje (objetivo)
    epsilon: float = 0.2     # Tolerancia de verificación (mm)
    noise_std: float = 0.0   # Ruido (desvío) de la heurística (mm)
    seed: Optional[int] = 42 # Semilla para reproducibilidad

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def verify(self, x: float) -> bool:
        """Devuelve True si la posición x cumple tolerancia de encastre."""
        return abs(x - self.A) <= self.epsilon

    def heuristic_estimate_delta(self, x: float) -> float:
        """
        Estima Δx = (A - x) con ruido gaussiano. Simula un 'relieve' o sensor.
        En la práctica podría venir de visión/sensado. Aquí es un proxy simple.
        """
        noise = self.rng.gauss(0.0, self.noise_std) if self.noise_std > 0 else 0.0
        return (self.A - x) + noise


# -----------------------------
#  Búsqueda EXHAUSTIVA (zig-zag)
# -----------------------------
def exhaustive_search(env: LineAssemblyEnv, B: float, step: float, max_steps: int = 10_000
                     ) -> Tuple[bool, float, List[float]]:
    """
    Recorre simétricamente desde B: B, B+step, B-step, B+2step, B-2step, ...
    Retorna (hallado, x_final, trayectoria)
    """
    k = 0
    path = []
    # Chequeo en B
    path.append(B)
    if env.verify(B):
        return True, B, path

    # Zig-zag incremental
    while len(path) < max_steps:
        k += 1
        for direction in (+1, -1):
            candidate = B + direction * k * step
            path.append(candidate)
            if env.verify(candidate):
                return True, candidate, path

    return False, path[-1], path


# -----------------------------
#  Búsqueda HEURÍSTICA (Greedy)
# -----------------------------
def heuristic_search_greedy(env: LineAssemblyEnv, B: float, max_iters: int = 10_000,
                            max_step: float = 1.0, gain: float = 1.0
                           ) -> Tuple[bool, float, List[float]]:
    """
    Greedy guiado por una estimación de Δx:
      x <- x + clip(gain * estimate, -max_step, +max_step)
    gain: cuánto confiar/multiplicar la estimación.
    max_step: tamaño máximo de micromovimiento por iteración.
    """
    x = B
    path = [x]
    if env.verify(x):
        return True, x, path

    for _ in range(max_iters):
        est = env.heuristic_estimate_delta(x)  # ≈ (A - x) + ruido
        move = max(-max_step, min(max_step, gain * est))
        # Si la estimación es muy chica, damos un pasito mínimo para no “oscilarnos”
        if abs(move) < 1e-6:
            move = math.copysign(min(max_step, 0.1), est if est != 0 else 1.0)
        x = x + move
        path.append(x)
        if env.verify(x):
            return True, x, path

    return False, x, path


# -----------------------------
#  Búsqueda HEURÍSTICA (A* 1D)
# -----------------------------
def heuristic_search_astar(env: LineAssemblyEnv, B: float, max_iters: int = 10_000,
                           step: float = 0.5) -> Tuple[bool, float, List[float]]:
    """
    A* en 1D sobre una discretización con paso 'step'.
    Vecinos: x - step, x + step.
    Heurística h(x) = |A - x| (admisible si no hay ruido).
    Nota: Si hay mucho ruido, Greedy puede rendir mejor por latencia.
    """
    # Discretizamos posiciones a una grilla para que A* tenga grafo finito (acotamos ventana).
    # Elegimos ventana +- W pasos alrededor de B (suficiente para hallar A).
    W = 400  # 400*0.5 = 200 mm de cada lado; ajustar según caso real
    def quantize(v: float) -> float:
        return round(v / step) * step

    start = quantize(B)
    goal = quantize(env.A)

    from heapq import heappush, heappop
    open_heap = []
    g = {start: 0.0}
    parent = {start: None}

    def h(x: float) -> float:
        return abs(env.A - x)

    heappush(open_heap, (g[start] + h(start), start))
    visited = set()
    iters = 0

    while open_heap and iters < max_iters:
        iters += 1
        f, x = heappop(open_heap)
        if env.verify(x):
            # reconstruir path
            path = []
            cur = x
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return True, x, path

        if x in visited:
            continue
        visited.add(x)

        # expandir vecinos
        for nx in (quantize(x - step), quantize(x + step)):
            # fuera de ventana
            if abs((nx - start) / step) > W:
                continue
            cost = g[x] + step
            if nx not in g or cost < g[nx]:
                g[nx] = cost
                parent[nx] = x
                heappush(open_heap, (g[nx] + h(nx), nx))

    # si no encontró, devolvemos lo mejor conocido
    # (reconstruimos al mejor x de g con menor h)
    best_x = min(g.keys(), key=lambda xx: h(xx))
    path = []
    cur = best_x
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return False, best_x, path


# -----------------------------
#  Utilidades de impresión
# -----------------------------
def summarize_run(name: str, ok: bool, x_final: float, path: List[float], env: LineAssemblyEnv):
    print(f"\n[{name}] resultado = {'ENCONTRADO' if ok else 'NO ENCONTRADO'}")
    print(f"  Posición final: {x_final:.3f} mm (objetivo A = {env.A:.3f} mm, ε = ±{env.epsilon:.3f} mm)")
    print(f"  Pasos / iteraciones: {len(path)-1}")
    if path:
        print(f"  Primera(s) posiciones: {', '.join(f'{p:.2f}' for p in path[:5])} ...")
        print(f"  Última(s) posiciones:  {', '.join(f'{p:.2f}' for p in path[-5:])}")


# -----------------------------
#  CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="TP2 - Búsqueda exhaustiva vs heurística en 1D (simulación)")
    parser.add_argument("--A", type=float, default=3.2, help="Posición objetivo A (mm)")
    parser.add_argument("--B", type=float, default=0.0, help="Posición inicial B (mm)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Tolerancia de encastre (mm)")
    parser.add_argument("--step", type=float, default=0.5, help="Paso ΔH para exhaustiva y/o A* (mm)")
    parser.add_argument("--noise", type=float, default=0.0, help="Desvío (std) de la heurística (mm)")
    parser.add_argument("--greedy", action="store_true", help="Usar heurística Greedy (por defecto se prueban ambas)")
    parser.add_argument("--astar", action="store_true", help="Usar heurística A* 1D")
    parser.add_argument("--exhaustive", action="store_true", help="Correr búsqueda exhaustiva (zig-zag)")
    parser.add_argument("--compare", action="store_true", help="Comparar exhaustiva vs Greedy y A*")
    parser.add_argument("--seed", type=int, default=42, help="Semilla RNG para reproducibilidad")
    args = parser.parse_args()

    env = LineAssemblyEnv(A=args.A, epsilon=args.epsilon, noise_std=args.noise, seed=args.seed)

    # Modos
    if args.compare or (not args.exhaustive and not args.greedy and not args.astar):
        print(">> Modo comparación (exhaustiva vs greedy vs A*)")
        ok1, xf1, path1 = exhaustive_search(env, B=args.B, step=args.step)
        summarize_run("EXHAUSTIVA", ok1, xf1, path1, env)

        ok2, xf2, path2 = heuristic_search_greedy(env, B=args.B, max_step=args.step, gain=1.0)
        summarize_run("HEURÍSTICA Greedy", ok2, xf2, path2, env)

        ok3, xf3, path3 = heuristic_search_astar(env, B=args.B, step=args.step)
        summarize_run("HEURÍSTICA A*", ok3, xf3, path3, env)
        return

    if args.exhaustive:
        ok, xf, path = exhaustive_search(env, B=args.B, step=args.step)
        summarize_run("EXHAUSTIVA", ok, xf, path, env)

    if args.greedy:
        ok, xf, path = heuristic_search_greedy(env, B=args.B, max_step=args.step, gain=1.0)
        summarize_run("HEURÍSTICA Greedy", ok, xf, path, env)

    if args.astar:
        ok, xf, path = heuristic_search_astar(env, B=args.B, step=args.step)
        summarize_run("HEURÍSTICA A*", ok, xf, path, env)


if __name__ == "__main__":
    main()
