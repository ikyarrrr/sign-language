import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28
dataset_size = 100

# Function to find the first available camera index
def find_camera():
    for index in range(5):  # Try up to 5 indices (0 to 4)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return -1  # Return -1 if no camera is found

camera_index = find_camera()
if camera_index == -1:
    print("Error: Could not find any available video stream or file")
    exit()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open video stream or file with index {camera_index}")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab frame")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab frame")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()