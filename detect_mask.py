import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load krege trained model
model = load_model("models/mask_detector_model.keras")

# Haarcascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

labels = ["Mask", "No Mask"]

cap = cv2.VideoCapture(0)

frame_count=0
with_cnt=0
without_cnt=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # frame_count += 1
    # if frame_count % 2 != 0:

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        face = np.expand_dims(face/255.0, axis=0)

        pred = model.predict(face)[0][0]

        # if pred < 0.3: 
        #     label = "Mask"
        # elif pred > 0.7: 
        #     label = "No Mask"
        # else:
        #     label = "Uncertain"

        label = labels[0] if pred < 0.5 else labels[1]
        if pred < 0.5:
            with_cnt += 1
        else:
            without_cnt += 1

        color = (0,255,0) if label=="Mask" else (0,0,255)
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

    cv2.imshow("Mask Detection", frame)

    print("With Mask:", with_cnt)
    print("Without Mask:", without_cnt)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

