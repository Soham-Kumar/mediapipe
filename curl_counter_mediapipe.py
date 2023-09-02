import cv2
import mediapipe as mp
import numpy as np

mp_drawing = (
    mp.solutions.drawing_utils
)  # Gives drawing utilities, used for visualizing poses
mp_pose = mp.solutions.pose  # Imports pose estimation model


# Function to calculate angle
def angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Variables for curls
left_counter = 0
right_counter = 0
l_stage = None
r_stage = None

# Make Detections
cap = cv2.VideoCapture(0)  # 0 for webcam, 1 for external webcam
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read webcam feed
        ret, frame = cap.read()

        # Recolor the image to RGB from BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = pose.process(image)
        image.flags.writeable = True

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            l_shoulder = [left_shoulder.x, left_shoulder.y]
            r_shoulder = [right_shoulder.x, right_shoulder.y]
            l_elbow = [left_elbow.x, left_elbow.y]
            r_elbow = [right_elbow.x, right_elbow.y]
            l_wrist = [left_wrist.x, left_wrist.y]
            r_wrist = [right_wrist.x, right_wrist.y]

            # Calculate angle
            left_angle = angle(l_shoulder, l_elbow, l_wrist)
            right_angle = angle(r_shoulder, r_elbow, r_wrist)

            # Curl counter logic
            if left_angle > 160:
                l_stage = "down"
            if left_angle < 30 and l_stage == "down":
                l_stage = "up"
                left_counter += 1

            if right_angle > 160:
                r_stage = "down"
            if right_angle < 30 and r_stage == "down":
                r_stage = "up"
                right_counter += 1

            # Write the number of curls to screen
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(
                image,
                "Left Reps",
                (15, 12),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(left_counter),
                (10, 60),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(image, (415, 0), (640, 73), (245, 117, 16), -1)
            cv2.putText(
                image,
                "Right Reps",
                (420, 12),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(right_counter),
                (420, 60),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        # Show to screen
        cv2.imshow("Webcam Feed", image)

        if cv2.waitKey(1) == ord("q"):
            break
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.release()
cv2.destroyAllWindows()
