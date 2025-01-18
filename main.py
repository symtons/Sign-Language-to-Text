import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque

# Load the trained model
model_path = "./models/sign_language_model2.keras"  # Update with your trained model's path
model = load_model(model_path)

# Define class labels
#class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space", "Delete", "Nothing"]
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to preprocess the hand image for the model
def preprocess_hand_image(image, target_size=(224, 224)):
    image_resized = cv2.resize(image, target_size)  # Resize to model input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    image_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    return image_batch

# Rolling predictions buffer
rolling_predictions = deque(maxlen=5)

# Function to map predictions to text
def map_to_text(predicted_class, text_buffer):
    if predicted_class == "Space":
        text_buffer += " "
    elif predicted_class == "Delete":
        text_buffer = text_buffer[:-1]  # Remove the last character
    elif predicted_class != "Nothing":
        text_buffer += predicted_class
    return text_buffer

# Start webcam feed
cap = cv2.VideoCapture(0)
text_buffer = ""  # To store the converted text

print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box of the hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Add padding to the bounding box
            padding = 30  # Increase padding for better cropping
            x_min, y_min = max(x_min - padding, 0), max(y_min - padding, 0)
            x_max, y_max = min(x_max + padding, w), min(y_max + padding, h)

            # Extract the hand region
            hand_image = frame[y_min:y_max, x_min:x_max]

            if hand_image.size > 0:
                # Preprocess the hand image
                processed_hand = preprocess_hand_image(hand_image)

                # Predict the gesture
                prediction = model.predict(processed_hand)
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Update rolling predictions
                rolling_predictions.append(predicted_class)
                final_prediction = max(set(rolling_predictions), key=rolling_predictions.count)

                # Debugging: Print predictions
                print(f"Predicted Gesture: {final_prediction}, Confidence: {confidence}")

                # Display prediction and confidence
                if confidence > 0.9:  # Apply confidence threshold
                    text_buffer = map_to_text(final_prediction, text_buffer)
                    cv2.putText(
                        frame, f"{final_prediction} ({confidence:.2f})",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )
                else:
                    cv2.putText(
                        frame, "Uncertain",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                    )

    # Display the converted text
    #cv2.putText(frame, f"Text: {text_buffer}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Sign Language to Text", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
