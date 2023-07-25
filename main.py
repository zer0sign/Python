import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def classify_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300)) 
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo 

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        class_name, confidence_score = classify_image(opencv_image)

        canvas.delete("result_text")
        canvas.delete("hint")

        result_text = f"Class: {class_name[2:]} - Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
        canvas.create_text(canvas_width // 2, canvas_height // 2, text=result_text, font=("Arial", 14), tags="result_text")

app = tk.Tk()
app.title("Image Classifier")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

initial_width = int(screen_width * 0.6)  
initial_height = int(screen_height * 0.6)  
app.geometry(f"{initial_width}x{initial_height}")

# Create a label to display the uploaded image
image_label = tk.Label(app)
image_label.pack(pady=10)

canvas_width = 300
canvas_height = 300
canvas = tk.Canvas(app, width=canvas_width, height=canvas_height)
canvas.pack(pady=5)


hint = canvas.create_text(canvas_width // 2, canvas_height // 2, text="Upload Image of cat or dog", font=("Arial", 14), tags="hint")

upload_button = tk.Button(app, text="Upload Image", command=open_image)
upload_button.pack(pady=5)


app.mainloop()
