import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Labels for the entire English alphabet and some additional words
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'hello'
}

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("800x600")

        # Layout configuration
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.panel = tk.Label(self.video_frame)
        self.panel.pack()

        self.prediction_label = tk.Label(self.info_frame, text="Prediction: ", font=("Courier", 20))
        self.prediction_label.pack(pady=10)

        self.confidence_label = tk.Label(self.info_frame, text="Confidence: ", font=("Courier", 20))
        self.confidence_label.pack(pady=10)

        self.sentence_label = tk.Label(self.info_frame, text="Sentence: ", font=("Courier", 20))
        self.sentence_label.pack(pady=10)

        self.suggestions_frame = tk.Frame(self.info_frame)
        self.suggestions_frame.pack(pady=10)

        self.suggestions = []
        self.suggestion_labels = []

        self.sentence = ""
        self.current_suggestion = None

        self.last_prediction_time = 0
        self.prediction_delay = 1.0  # Delay in seconds
        self.stable_gesture_time = 2.0  # Time in seconds to hold a gesture for it to be considered stable
        self.last_gesture_time = 0
        self.current_gesture = None

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.root.quit()
            return

        self.process_frame()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            self.root.after(1000, self.process_frame)
            return

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Extract hand landmarks and normalize
            data_aux = []

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x)
                    data_aux.append(y)

            while len(data_aux) < 84:
                data_aux.extend([0, 0])

            assert len(data_aux) == 84, "Ensure data_aux matches the expected feature count"

            # Make prediction if enough time has passed
            if current_time - self.last_prediction_time > self.prediction_delay:
                self.last_prediction_time = current_time
                prediction_proba = model.predict_proba([np.asarray(data_aux)])[0]
                sorted_indices = np.argsort(prediction_proba)[::-1]  # Indices of probabilities sorted in descending order

                # Print the raw predictions and indices for debugging
                print("Predicted probabilities:", prediction_proba)
                print("Sorted indices:", sorted_indices)

                # Select top N predictions
                top_n = 5
                top_indices = sorted_indices[:top_n]
                top_predictions = {}
                for i in top_indices:
                    if int(i) in labels_dict:
                        top_predictions[labels_dict[int(i)]] = prediction_proba[i]
                    else:
                        print(f"Warning: Model predicted an index {i} which is not in the labels_dict")

                if top_predictions:
                    # Determine the most probable character
                    predicted_character = max(top_predictions, key=top_predictions.get)
                    confidence_score = top_predictions[predicted_character]

                    # Update GUI with prediction and confidence
                    self.prediction_label.config(text=f"Prediction: {predicted_character}")
                    self.confidence_label.config(text=f"Confidence: {confidence_score:.2f}")

                    # Check if the predicted character has been consistent for a duration
                    if self.current_gesture == predicted_character:
                        if current_time - self.last_gesture_time > self.stable_gesture_time:
                            self.sentence += predicted_character
                            self.sentence_label.config(text=f"Sentence: {self.sentence}")
                            self.current_gesture = None  # Reset current gesture
                    else:
                        self.current_gesture = predicted_character
                        self.last_gesture_time = current_time

                    # Update suggestions list
                    self.update_suggestions_list(top_predictions)

        # Convert frame to ImageTk format and update GUI
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        self.root.after(10, self.process_frame)

    def update_suggestions_list(self, top_predictions):
        # Clear previous suggestions
        for label in self.suggestion_labels:
            label.destroy()
        self.suggestion_labels = []

        # Sort predictions by probability in descending order
        sorted_suggestions = sorted(top_predictions.items(), key=lambda x: x[1], reverse=True)

        # Display new suggestions
        for i, (suggestion, probability) in enumerate(sorted_suggestions):
            label = tk.Label(self.suggestions_frame, text=f"{i+1}: {suggestion} ({probability:.2f})", font=("Courier", 15), cursor="hand2")
            label.pack(pady=2, anchor='w')
            label.bind("<Button-1>", self.on_suggestion_click)
            self.suggestion_labels.append(label)

    def on_suggestion_click(self, event):
        clicked_text = event.widget.cget("text")
        selected_character = clicked_text.split(": ")[1].split(" ")[0]
        self.sentence += selected_character
        self.sentence_label.config(text=f"Sentence: {self.sentence}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    app.run()
