xml = "C:/Users/mndar/PycharmProjects/4road/road.xml"
video = "C:/Users/mndar/PycharmProjects/4road/test.mp4"
# video="C:/Users/mndar/PycharmProjects/4road/test1.mp4"


import cv2

# Load the Haar Cascade file
cascade = cv2.CascadeClassifier(xml)

# Open the video file
video = cv2.VideoCapture("C:/Users/mndar/PycharmProjects/4road/test.mp4")

# Define the ROI coordinates (x, y, w, h)
roi = (600, 600, 400, 300)

# Define a font for displaying the object count
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
# Loop over each frame in the video
while True:
    # Read the frame
    ret, frame = video.read()

    # Break the loop if we've reached the end of the video
    if not ret:
        break

    # Define the ROI
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using the Haar Cascade
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected objects and display the object count

    for (x, y, w, h) in objects:
        cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        count += 1
    cv2.putText(roi_frame, f"Objects detected: {count}", (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video frame
    cv2.imshow('Video', frame)
    cv2.waitKey(50)
    # Wait for a key press and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()
