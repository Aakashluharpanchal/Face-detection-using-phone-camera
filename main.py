import cv2

# Replace 'your_phone_ip_address' with the IP address of your phone running the IP Webcam app
# Replace '8080' with your camera port number which is usually 8080
# Best would be download IP Webcam app in an android phone as soon as the server is started IP address and port number
# are displayed on screen

url = 'http://191.168.1.146:8080/video'

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the IP Webcam video stream
cap = cv2.VideoCapture(url)

# Check if the camera stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera stream")
    exit()
    exit()

# Define the desired width and height for resizing
desired_width = 640
desired_height = 480

# Read and display frames from the camera stream
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame from camera stream")
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    rotate_frame = cv2.rotate(resized_frame,cv2.ROTATE_90_CLOCKWISE)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(rotate_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(rotate_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Camera Feed', rotate_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
