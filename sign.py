import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the trained model
model_path = "./models/sign_language_model.keras"  # Update with your trained model's path
model = load_model(model_path)

# Define class labels
class_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space", "Delete", "Nothing"
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to preprocess the hand image for the model
def preprocess_hand_image(image, target_size=(224, 224)):
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch

# Function to map predictions to text
def map_to_text(predicted_class, text_buffer):
    if predicted_class == "Space":
        text_buffer += " "
    elif predicted_class == "Delete":
        text_buffer = text_buffer[:-1]
    elif predicted_class != "Nothing":
        text_buffer += predicted_class
    return text_buffer

# Tkinter Application Class
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text")
        self.root.geometry("800x600")

        # Rolling predictions buffer
        self.rolling_predictions = deque(maxlen=5)

        # Text buffer
        self.text_buffer = ""

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Webcam feed display
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Text display
        self.text_label = tk.Label(root, text="Text: ", font=("Helvetica", 16), anchor="w")
        self.text_label.pack(fill="x")

        # Buttons
        self.save_button = tk.Button(root, text="Save Text", command=self.save_text)
        self.save_button.pack(side="left", padx=10)

        self.clear_button = tk.Button(root, text="Clear Text", command=self.clear_text)
        self.clear_button.pack(side="left", padx=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app)
        self.exit_button.pack(side="right", padx=10)

        # Start video stream
        self.update_video_stream()

    def save_text(self):
        # Save the text buffer to a file
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.text_buffer)

    def clear_text(self):
        # Clear the text buffer
        self.text_buffer = ""
        self.text_label.config(text="Text: ")

    def exit_app(self):
        # Release resources and close the app
        self.cap.release()
        self.root.destroy()

    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands in the frame
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get bounding box
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                    # Add padding
                    padding = 30
                    x_min, y_min = max(x_min - padding, 0), max(y_min - padding, 0)
                    x_max, y_max = min(x_max + padding, w), min(y_max + padding, h)

                    # Extract hand region
                    hand_image = frame[y_min:y_max, x_min:x_max]

                    if hand_image.size > 0:
                        # Preprocess the hand image
                        processed_hand = preprocess_hand_image(hand_image)

                        # Predict the gesture
                        prediction = model.predict(processed_hand)
                        predicted_class = class_labels[np.argmax(prediction)]
                        confidence = np.max(prediction)

                        # Update rolling predictions
                        self.rolling_predictions.append(predicted_class)
                        final_prediction = max(set(self.rolling_predictions), key=self.rolling_predictions.count)

                        # Map prediction to text buffer
                        if confidence > 0.9:
                            self.text_buffer = map_to_text(final_prediction, self.text_buffer)

            # Update text display
            self.text_label.config(text=f"Text: {self.text_buffer}")

            # Display the frame on the canvas
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk

        # Continue updating
        self.root.after(10, self.update_video_stream)


# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
