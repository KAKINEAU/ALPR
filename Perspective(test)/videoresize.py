import cv2

# Open a video stream from the default camera (webcam)
cap = cv2.VideoCapture(1)
# Set the video resolution to the maximum resolution
cap.set(3, 1280)
cap.set(4, 720)
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # Get the width and height of the frame
    width = int(cap.get(3))
    height = int(cap.get(4))

    print("Frame size: {}x{}".format(width, height))

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()