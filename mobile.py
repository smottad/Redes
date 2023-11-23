import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

class ObjectIdentificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Identificación de Objetos")

        # Load the pre-trained MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')

        # Create and place widgets
        self.load_button = tk.Button(root, text="Cargar Imagen", command=self.load_image)
        self.load_button.pack(pady=10)

        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        self.result_label = tk.Label(root, text="Resultados de la predicción:")
        self.result_label.pack(pady=10)

        # Labels to display top three predictions and their confidence scores
        self.prediction_labels = [
            tk.Label(root, text=f"Predicción 1: "),
            tk.Label(root, text=f"Predicción 2: "),
            tk.Label(root, text=f"Predicción 3: ")
        ]

        for label in self.prediction_labels:
            label.pack()

        # Restart button
        self.restart_button = tk.Button(root, text="Reiniciar", command=self.restart)
        self.restart_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            print(f"Imagen seleccionada: {self.image_path}")

            # Load and display the image
            image = Image.open(self.image_path)
            image = image.resize((224, 224))
            image = ImageTk.PhotoImage(image)
            self.panel.config(image=image)
            self.panel.image = image

            # Predict the image
            self.predict_image(self.image_path)

    def predict_image(self, image_path):
        try:
            # Load and preprocess the image for prediction
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make predictions
            predictions = self.model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Display the top three predictions and their confidence scores
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                self.prediction_labels[i].config(text=f"Predicción {i + 1}: {label} (Confianza: {score:.2%})")

        except Exception as e:
            print(f"Error during prediction: {str(e)}")

    def restart(self):
        # Clear the image, result labels, and image path
        self.panel.config(image="")
        self.result_label.config(text="Resultados de la predicción:")
        for label in self.prediction_labels:
            label.config(text="")
        self.image_path = ""

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectIdentificationApp(root)
    root.mainloop()
