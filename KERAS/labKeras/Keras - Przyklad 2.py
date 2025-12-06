# Ładowanie potrzebych modułów
# MNIST - zbiór obrazów z odręcznie pisanymi cyframi od 0 do 9
# Sequential- model sekwencyjny sieci neuronowej
try:
    from keras.api.datasets import mnist
    from keras.api.models import Sequential, load_model, model_from_json
    from keras.api import layers
    from keras.api.utils import to_categorical
except:
    from keras.datasets import mnist
    from keras.models import Sequential, load_model, model_from_json
    from keras import layers
    from keras.utils import to_categorical
from matplotlib import pyplot as plt

# Wczytywanie danych
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Przekształcanie wielkości obrazów do 28x28x1 pixel oraz ich normalizacja
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Pobranie i stworzenie listy klas dla danych
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Tworzenie modelu sieci
model = Sequential()

# Dodanie pierwszej warsty konwolucyjnej złożonej z 32 kerneli o wielkości 3x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Dodanie warstwy zmiejszającej wielkość powstałych obrazów z warstwy konwolucyjnej
model.add(layers.MaxPooling2D((2, 2)))

# Dodanie drugiej warstwy konwolucyjnej
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Dodanie warstwy spłaszczającej dane 2D do 1D
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

# Dodanie warstwy gęstej odpowiedzialnej za klasę - liczbla neuronow = liczba klas
model.add(layers.Dense(10, activation='softmax'))

#Kompilacja modelu
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Uczenie modelu danymi
# epoch - liczba iteracji
# batch_size - liczba elementów z danych treningowych branych podczas pojedyńczego przejścia funkcji uczącej
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5, batch_size=64, verbose=1)

print(history.history)
# wyświetlenie wykresu przedstawiającego historię uczenia sieci
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# zapis i odczyt

# 1. Zapis całego modelu (architektura + wagi) do jednego pliku H5
model.save("model2_full.h5")
print("Zapisano pełny model do pliku h5")

# 2. Zapis osobno: architektura do JSON, wagi do H5
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model2_weights.weights.h5")
print("Zapisano architekturę (JSON) i wagi (H5)")
