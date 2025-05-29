import cv2

print(f"OpenCV version: {cv2.__version__}")

# Try to open camera at index 0
cap0 = cv2.VideoCapture(0)
if cap0.isOpened():
    print("Successfully opened camera at index 0 (/dev/video0)")
    ret, frame = cap0.read()
    if ret:
        print("Successfully read a frame from camera 0.")
    else:
        print("Could not read a frame from camera 0, even though it's opened.")
    cap0.release()
else:
    print("Failed to open camera at index 0 (/dev/video0)")

print("-" * 30)

# Try to open camera at index 1
cap1 = cv2.VideoCapture(1)
if cap1.isOpened():
    print("Successfully opened camera at index 1 (/dev/video1)")
    ret, frame = cap1.read()
    if ret:
        print("Successfully read a frame from camera 1.")
    else:
        print("Could not read a frame from camera 1, even though it's opened.")
    cap1.release()
else:
    print("Failed to open camera at index 1 (/dev/video1)")

print("-" * 30)

# Explicitly try opening by path (sometimes more reliable)
cap_path0 = cv2.VideoCapture("/dev/video0")
if cap_path0.isOpened():
    print("Successfully opened camera by path /dev/video0")
    cap_path0.release()
else:
    print("Failed to open camera by path /dev/video0")

print("-" * 30)

cap_path1 = cv2.VideoCapture("/dev/video1")
if cap_path1.isOpened():
    print("Successfully opened camera by path /dev/video1")
    cap_path1.release()
else:
    print("Failed to open camera by path /dev/video1")