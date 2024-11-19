import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Визначаємо параметри бета-розподілу
alpha, beta = 2, 5


# Функція, що описує цільовий розподіл (щільність бета-розподілу)
def target_distribution(x):
    if 0 <= x <= 1:
        return stats.beta.pdf(x, alpha, beta)
    else:
        return 0


# Алгоритм Метрополіса-Гастінгса
def metropolis_hastings(target_distribution, iterations, proposal_width=0.5):
    samples = []
    current_sample = np.random.rand()  # Початкове значення з інтервалу [0, 1]
    for _ in range(iterations):
        # Пропонуємо нове значення з нормального розподілу навколо поточного значення
        proposed_sample = current_sample + np.random.normal(0, proposal_width)

        # Обчислюємо ймовірність прийняття
        acceptance_probability = min(1, target_distribution(proposed_sample) / target_distribution(current_sample))

        # Приймаємо чи відхиляємо нове значення
        if np.random.rand() < acceptance_probability:
            current_sample = proposed_sample

        # Додаємо поточне значення до вибірки
        samples.append(current_sample)
    return np.array(samples)


# Змоделюємо 1000 значень за допомогою алгоритму Метрополіса-Гастінгса
iterations = 1000
samples = metropolis_hastings(target_distribution, iterations)

# Побудова графіку реалізації
plt.figure(figsize=(10, 6))
plt.plot(samples, marker='o', linestyle='-', markersize=2)
plt.title('Графік реалізації алгоритму Метрополіса-Гастінгса')
plt.xlabel('Ітерація')
plt.ylabel('Значення')
plt.grid(True)
plt.show()

# Побудова гістограми зразків
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7, edgecolor='black')
plt.title('Гістограма зразків та щільність бета-розподілу')
plt.xlabel('Значення')
plt.ylabel('Частота')

# Додавання щільності бета-розподілу для порівняння
x = np.linspace(0, 1, 1000)
plt.plot(x, stats.beta.pdf(x, alpha, beta), 'r-', lw=2, label='Бета-розподіл')
plt.legend()
plt.grid(True)
plt.show()
