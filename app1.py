import cv2
import pygame
from ultralytics import YOLO
import threading
import winsound
import os

# Initialize Pygame
pygame.init()

# Set up display dimensions and caption
screen_width, screen_height = 780, 620
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Y-BIN")

# Load YOLOv8 model (make sure the path is correct)
model = YOLO('yolo11s.pt')  # Adjust the model path as necessary

# Define a list of food items based on the YOLO COCO labels
food_items = ["apple", "banana", "orange", "carrot", "pizza", "cake", "sandwich","hand"]

# Initialize video capture (use 0 for webcam or specify a video file)
cap = cv2.VideoCapture(0)

# Function to play the alert sound in a separate thread
def play_alert_sound():
    sound_path = "alert-33762.mp3"
    if os.path.exists(sound_path):
        winsound.PlaySound(sound_path, winsound.SND_FILENAME)
    else:
        print(f"Alert sound file '{sound_path}' not found.")

# Main loop
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to fit Pygame display
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Use YOLO model to perform inference
    results = model(frame)

    # Initialize flag to track if any non-food item is detected
    non_food_detected = False

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        classes = result.boxes.cls  # Class IDs
        confidences = result.boxes.conf  # Confidences for each box

        for i, cls in enumerate(classes):
            label = model.names[int(cls)]  # Get the label of the detected object
            confidence = confidences[i].item()

            # Set color based on whether it's a food item
            color = (0, 255, 0) if label in food_items else (255, 0, 0)

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if the detected item is not in the food list
            if label not in food_items:
                non_food_detected = True

    # Play the alert sound if non-food items are detected and ensure it's played only once
    if non_food_detected:
        cv2.putText(frame, "Non-Food Item Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not threading.active_count() > 1:  # Check if sound is already playing
            threading.Thread(target=play_alert_sound).start()

    # Convert the OpenCV image (BGR) to Pygame format (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    frame_surface = pygame.transform.rotate(frame_surface, -90)  # Rotate if needed
    frame_surface = pygame.transform.flip(frame_surface, True, False)

    # Display frame in Pygame window
    screen.blit(frame_surface, (0, 0))
    pygame.display.update()

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Press 'q' to quit
                running = False

cap.release()
pygame.quit()
