
# Implementacja obsługi ładowania i predykcji modelu

try:
    from keras.api.models import model_from_json
    from keras.api.models import load_model
except:
    from keras.models import model_from_json
    from keras.models import load_model
import cv2
import numpy as np

from PyQt6.QtGui import QImage

def qimage_to_array(image: QImage):
    """
    Funkcja konwertująca obiekt QImage do numpy array
    """
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    numpy_array = np.array(ptr).reshape(image.height(), image.width(), 1)

    # wykorzystanie bibloteki OpenCV do wyświetlenia obrazu po konwersji
    cv2.imshow('Check if the function works!', numpy_array)
    return numpy_array
    

def predict(image: QImage, model):
    if model is None:
        return None

    # 1. QImage -> numpy (wysokość, szerokość, 1) w skali szarości
    numpy_array = qimage_to_array(image)

    # 2. Zmiana rozmiaru do 28x28 – tak jak w MNIST
    numpy_array = cv2.resize(numpy_array, (28, 28))

    # 3. Normalizacja do zakresu [0, 1]
    numpy_array = numpy_array.astype("float32") / 255.0

    # 4. Dopasowanie kształtu do wejścia modelu
    #    - dla sieci gęstej:  (None, 784)
    #    - dla sieci CNN:     (None, 28, 28, 1)
    input_shape = model.input_shape

    if len(input_shape) == 2:
        # Model z warstwami gęstymi
        prepared = numpy_array.reshape(1, 28 * 28)
    elif len(input_shape) == 4:
        # Model konwolucyjny
        prepared = numpy_array.reshape(1, 28, 28, 1)
    else:
        # Awaryjnie
        prepared = np.expand_dims(numpy_array, axis=0)

    # 5. Predykcja
    predictions = model.predict(prepared)
    predicted_class = int(np.argmax(predictions[0]))

    # (Opcjonalnie) podgląd obrazu po przeskalowaniu
    # cv2.imshow('Check if the function works!!', numpy_array)
    # cv2.waitKey(1)

    return predicted_class


def get_model():
    model_path = "model2_full.h5"

    try:
        model = load_model(model_path)
        print(f"Załadowano model z pliku: {model_path}")
    except Exception as exc:
        print(f"Nie udało się załadować modelu z pliku {model_path}: {exc}")
        model = None

    return model