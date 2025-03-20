import dlib
import cv2

camera = cv2.VideoCapture(1)

file_path = "mmod_human_face_detector.dat"

detector = dlib.cnn_face_detection_model_v1(file_path)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_downscaled = cv2.resize(frame_gray, (frame_gray.shape[1] // 2, frame_gray.shape[0] // 2))

    detections = detector(frame_downscaled, 1)

    for face in detections:
        x, y, w, h = (face.rect.left() * 2,
                      face.rect.top() * 2,
                      face.rect.right() * 2,
                      face.rect.bottom() * 2
                      )
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)


    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
