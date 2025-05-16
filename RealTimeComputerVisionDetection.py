import cv2
import mediapipe as mp
import time
import numpy as np
import deepface as DeepFace

# Initialize MediaPipe solutions
mpFace = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# Set up models
faceDetection = mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5)
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=2)
pose = mpPose.Pose(static_image_mode=False, model_complexity=1)
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Helper functions
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle

def eye_aspect_ratio(eye_landmarks):
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Open webcam
cap = cv2.VideoCapture(0)

# Set up video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      20.0,
                      (frame_width, frame_height))

# Initialize variables
pTime = 0
previous_frame = None
recording = False
frame_count = 0
emotion_frame_skip = 0  # To control how often we run emotion detection

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh (468 landmarks)
    mesh_results = faceMesh.process(imgRGB)
    if mesh_results.multi_face_landmarks:
        for faceLms in mesh_results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                frame, faceLms,
                mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
            
            # Distance estimation
            face_width = abs(faceLms.landmark[234].x - faceLms.landmark[454].x)
            distance = 50 / face_width  # approximate formula
            cv2.putText(frame, f"Distance: {int(distance)}cm", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Eye blink detection
            right_eye = np.array([[faceLms.landmark[p].x, faceLms.landmark[p].y] 
                                 for p in [33, 160, 158, 133, 153, 144]])
            ear = eye_aspect_ratio(right_eye)
            if ear < 0.2:
                cv2.putText(frame, "Blink Detected!", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Emotion detection (run every 30 frames to improve performance)
    if emotion_frame_skip % 30 == 0:
        try:
            emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if emotion:
                dominant_emotion = emotion[0]['dominant_emotion']
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
    emotion_frame_skip += 1

    # Pose detection
    pose_results = pose.process(imgRGB)
    if pose_results.pose_landmarks:
        mpDraw.draw_landmarks(frame, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        # Calculate angle of right elbow
        landmarks = pose_results.pose_landmarks.landmark
        right_shoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mpPose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mpPose.PoseLandmark.RIGHT_WRIST]
        
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        cv2.putText(frame, f"R Elbow Angle: {int(angle)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hand detection and gesture recognition
    hand_results = hands.process(imgRGB)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            
            # Get hand landmarks
            landmarks = handLms.landmark
            
            # Check if index finger is up
            if landmarks[8].y < landmarks[7].y < landmarks[6].y:
                cv2.putText(frame, "Index Finger Up!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Motion detection
    if previous_frame is not None:
        frame_diff = cv2.absdiff(previous_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        motion = np.mean(frame_diff)
        if motion > 20:
            cv2.putText(frame, "Motion Detected!", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Recording status
    if recording:
        cv2.putText(frame, "Recording...", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Full Body Detection", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
    elif key == ord('s'):
        cv2.imwrite(f'screenshot_{frame_count}.jpg', frame)
        frame_count += 1

    # Record frame if recording is enabled
    if recording:
        out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()