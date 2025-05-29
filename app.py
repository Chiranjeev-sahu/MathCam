from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import os
import handTrack as ht
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()
app = Flask(__name__)

# --- Global Variables ---
BRUSH_THICKNESS = 15 # Renamed for clarity as a constant
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# DEFAULT DRAWING COLOR IS RED (BGR format: Blue=0, Green=0, Red=255)
drawColor = (0, 0, 255)

xp, yp = 0, 0 # Previous drawing coordinates
# Initialize imgCanvas with the TARGET dimensions
imgCanvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), np.uint8)

# --- Hand Detector Setup (Global) ---
detector = ht.handDetector(detectionCon=0.85) # Using 0.85 detection confidence
print("Hand detector initialized globally.")

# --- Gemini Setup (Global) ---
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    api_key = api_key.strip()
    genai.configure(api_key=api_key)
else:
    print("Warning: GOOGLE_API_KEY not found. AI functionality will not work.")


def gen_frames():
    global xp, yp, imgCanvas

    print("Attempting to open camera in gen_frames()...")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        # ... (error handling for camera not opening) ...
        print("Error: Could not open video stream in gen_frames()")
        error_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        cv.putText(error_frame, "CAMERA ERROR", (50, TARGET_HEIGHT // 2), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        ret, buffer = cv.imencode(".jpg", error_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    print("Camera opened successfully in gen_frames()")

    frame_count = 0

    try:
        while True:
            success, img_original_from_cam = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break
            
            img_resized = cv.resize(img_original_from_cam, (TARGET_WIDTH, TARGET_HEIGHT))
            img_flipped = cv.flip(img_resized, 1) # Flip horizontally for mirror view
            
            # --- CORRECTED USAGE OF findHands ---
            # Let findHands draw directly on the image we will use.
            # The 'img' variable will now have the skeleton if hands are found.
            img_with_skeleton = detector.findHands(img_flipped.copy()) # Pass a copy if findHands modifies, or ensure it returns the modified img
                                                                      # findHands in handTrack.py returns the modified image.
            
            # Now use img_with_skeleton to get landmarks
            lmList = detector.findPosition(img_with_skeleton, draw=False) # draw=False here is fine as findHands already drew
            # --- --- --- --- --- --- --- --- --- --- ---

            # The main image we'll draw on and display will be img_with_skeleton
            # If you need to draw the brush tip on an image that *already* has the skeleton:
            current_display_img = img_with_skeleton 
            # If findHands didn't modify the input, and returned a new image, assign it:
            # current_display_img = result_from_find_hands 

            frame_count += 1

            if len(lmList) != 0: # If a hand is detected
                if frame_count % 30 == 0: 
                    print(f"Hand Detected! lmList length: {len(lmList)}")

                x1, y1 = lmList[8][1:] # Index finger tip

                fingers = detector.fingersUp() 
                if frame_count % 30 == 0:
                    print(f"Fingers state: {fingers}") 

                # --- GESTURE LOGIC ---
                if (fingers[0] and fingers[1] and
                    not fingers[2] and not fingers[3] and not fingers[4]):
                    cv.imwrite("saved_canvas.jpg", imgCanvas)
                    if frame_count % 30 == 0: print("SAVE gesture: Canvas saved.")
                    cv.putText(current_display_img, "SAVED!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif fingers[1] and not fingers[2]: 
                    if frame_count % 30 == 0: print(f"DRAWING MODE! Color: {drawColor}, Pos: ({x1},{y1})")
                    cv.circle(current_display_img, (x1, y1), BRUSH_THICKNESS // 2 + 5, drawColor, cv.FILLED)
                    if xp == 0 and yp == 0: xp, yp = x1, y1   
                    cv.line(current_display_img, (xp, yp), (x1, y1), drawColor, BRUSH_THICKNESS) # Draw on display img
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, BRUSH_THICKNESS) # Draw on canvas
                    xp, yp = x1, y1
                else: 
                    xp, yp = 0, 0
            else: 
                if frame_count % 60 == 0: print("No hand detected.")
                xp, yp = 0, 0


            # --- IMAGE COMPOSITING ---
            # current_display_img already has the live feed + skeleton + brush tip
            imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
            _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
            imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
            
            # Apply mask to current_display_img
            final_img = cv.bitwise_and(current_display_img, imgInv)
            # Overlay the drawing canvas
            final_img = cv.bitwise_or(final_img, imgCanvas)

            # --- Encode and Yield ---
            ret, buffer = cv.imencode(".jpg", final_img) # Encode the final composite image
            if not ret:
                print("Error: Could not encode frame.")
                continue
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    finally:
        print("gen_frames() loop ended. Releasing camera.")
        if 'cap' in locals() and cap.isOpened():
            cap.release()

# --- Flask Routes ---
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/gemini")
def gemini_route():
    if not genai.API_KEY: # Using the check you had
        return "Error: Gemini API key not configured.", 500
    try:
        if not os.path.exists("saved_canvas.jpg"):
            return "Error: No canvas image found. Save first (Thumb+Index up).", 404

        sample_file = genai.upload_file(path="saved_canvas.jpg", display_name="Maths Question")
        model = genai.GenerativeModel(model_name="gemini-1.5-pro") # Or your preferred model
        response = model.generate_content(
            [
                sample_file,
                "In the picture you have been provided a matrix question/equation. Please solve it and give numerical answer. First write the final solution then write the explanation. Give plain text response only.",
            ]
        )
        return response.text
    except Exception as e:
        print(f"Error in /gemini route: {e}")
        return f"An error occurred while processing with Gemini: {str(e)}", 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)