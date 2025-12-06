import numpy as np
from PIL import Image

try:
    from keras.api.models import load_model
except:
    from keras.models import load_model


def load_and_prepare_image(image_path):

    # 1. Wczytanie obrazu
    img = Image.open(image_path)

    # 2. Konwersja do skali szarości (L = 8-bit grayscale)
    img = img.convert("L")

    # 3. Zmiana rozmiaru na 28x28 (taki rozmiar ma MNIST)
    img = img.resize((28, 28))

    # 4. Zamiana na tablicę numpy
    img_array = np.array(img).astype("float32")

    # 5. Spłaszczenie (28x28 -> 784) i normalizacja do [0,1]
    img_array = img_array.reshape(1, 28 * 28) / 255.0

    return img_array


def main():
    # Ścieżka do wytrenowanego modelu z Zadania 2
    model_path = "model1_full.h5"

    # Ścieżka do obrazu, który chcemy sklasyfikować
    image_path = "cyfra4.png"

    # 1. Wczytanie modelu
    model = load_model(model_path)
    print(f"Załadowano model z pliku: {model_path}")

    # 2. Przygotowanie obrazu
    x = load_and_prepare_image(image_path)
    print(f"Załadowano i przetworzono obraz: {image_path}")

    # 3. Predykcja
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print("Rozkład prawdopodobieństw (0-9):")
    print(predictions[0])

    print(f"\nPrzewidywana cyfra: {predicted_class}")
    print(f"Pewność modelu: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
