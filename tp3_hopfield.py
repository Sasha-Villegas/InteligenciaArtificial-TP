
"""
TP3 - Prototipo de Red de Hopfield (desde cero)
Autor: Sasha Villegas Basso

Objetivo:
Implementar un modelo de Hopfield simple para reconocer patrones binarios (imágenes pequeñas)
y mostrar cómo la red recupera un patrón correcto a partir de una versión ruidosa.

Requisitos: solo numpy.
"""

import numpy as np

# ---------------------------
# 1. Funciones base del modelo
# ---------------------------

def train_hopfield(patterns):
    """
    Entrena la red de Hopfield a partir de una lista de patrones binarios.
    Aplica la regla de Hebb: W = sum(p.T * p) - nI
    """
    num_neurons = patterns[0].size
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        p = p.reshape(num_neurons, 1)
        W += np.dot(p, p.T)
    # eliminar auto-conexiones
    np.fill_diagonal(W, 0)
    W /= num_neurons
    return W


def update(state, W, asynchronous=True):
    """
    Actualiza los estados de las neuronas.
    Si asynchronous=True, actualiza en orden aleatorio (más biológico).
    """
    new_state = state.copy()
    indices = np.arange(len(state))
    if asynchronous:
        np.random.shuffle(indices)
    for i in indices:
        net = np.dot(W[i], new_state)
        new_state[i] = 1 if net >= 0 else -1
    return new_state


def recall(W, input_pattern, steps=5):
    """
    Intenta recuperar un patrón a partir de uno distorsionado.
    Itera varias veces hasta estabilizarse.
    """
    s = input_pattern.copy()
    for _ in range(steps):
        s = update(s, W)
    return s


def print_pattern(p, shape):
    """
    Muestra el patrón binario en consola (1=█, -1=·)
    """
    img = p.reshape(shape)
    for row in img:
        print("".join("█" if x == 1 else "·" for x in row))
    print()


# ---------------------------
# 2. Ejemplo de uso
# ---------------------------

if __name__ == "__main__":
    # Definimos patrones simples (letras pequeñas 3x3)
    A = np.array([[1, -1, 1],
                  [1,  1, 1],
                  [1, -1, 1]])

    X = np.array([[1, 1, -1],
                  [1, 1, -1],
                  [1, 1, 1]])

    patterns = [A.flatten(), X.flatten()]

    # Entrenamiento
    W = train_hopfield(patterns)
    print("=== Red entrenada con 2 patrones (A y X) ===")

    print("Patrón A original:")
    print_pattern(A.flatten(), (3, 3))

    # --- Caso 1: Ruido moderado (recuperación exitosa) ---
    noisy_A = A.flatten().copy()
    flip_indices = np.random.choice(len(noisy_A), 3, replace=False)
    noisy_A[flip_indices] *= -1

    print("Patrón A con ruido moderado:")
    print_pattern(noisy_A, (3, 3))

    recovered_A = recall(W, noisy_A, steps=6)
    print("Patrón A recuperado:")
    print_pattern(recovered_A, (3, 3))
    print("¿Recuperación correcta?", "Sí ✅" if np.array_equal(recovered_A, A.flatten()) else "No ❌")

    # --- Caso 2: Ruido excesivo (recuperación fallida) ---
    noisy_A2 = A.flatten().copy()
    flip_indices2 = np.random.choice(len(noisy_A2), 6, replace=False)  # 6 de 9 bits cambiados
    noisy_A2[flip_indices2] *= -1

    print("\nPatrón A con ruido excesivo:")
    print_pattern(noisy_A2, (3, 3))

    recovered_A2 = recall(W, noisy_A2, steps=6)
    print("Patrón recuperado (ruido alto):")
    print_pattern(recovered_A2, (3, 3))

    # Comprobamos si la red recuperó A, X o ninguno
    if np.array_equal(recovered_A2, A.flatten()):
        print("Resultado: recuperó A ✅")
    elif np.array_equal(recovered_A2, X.flatten()):
        print("Resultado: confundió con X ⚠️")
    else:
        print("Resultado: no convergió a un patrón válido ❌")

