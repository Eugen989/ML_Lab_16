import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[180, 70, 43],
              [190, 70, 46],
              [173, 67, 42],
              [170, 60, 40],
              [169, 80, 40],
              [156, 40, 36]])
y = np.array([1, 1, 1, 0, 0, 0])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')