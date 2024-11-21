import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import torch

# Kiểm tra xem CUDA có sẵn không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Khởi tạo mô hình YOLO
model = YOLO("D:\\pythonlearn\\runs\\detect\\train3\\weights\\best.pt")
model.to(device)

# Khởi tạo camera
video_capture = cv2.VideoCapture(0)

# Load và mã hóa khuôn mặt biết trước
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    images = [
        "C:\\Users\\DELL\\Pictures\\Camera Roll\\WIN_20241007_17_13_42_Pro.jpg"
        ]
    
    for img_path in images:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append("Tuong")
    images1 = [        
    "C:\\Users\\DELL\\Pictures\\Camera Roll\\WIN_20241113_14_37_34_Pro.jpg"]
    for img_path in images1:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append("Anh Kiet")
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {img_path}")
    images2 = [        
    "C:\\Users\\DELL\\Downloads\\Hinh-TQVinh-VINH-TRUONG-QUANG.jpg"]
    for img_path in images2:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append("Mr.Vinh")
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {img_path}")
    
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()
known_face_encodings_gpu = torch.tensor(known_face_encodings).to(device)

# Hàm để xử lý frame
def process_frame(frame):
    # Lật frame theo chiều ngang
    frame = cv2.flip(frame, 1)
    
    frame_resized = cv2.resize(frame, (640,640))
    
    results = model(frame_resized, conf=0.25)
    boxes = results[0].boxes.xywh.cpu().numpy()

    for box in boxes:
        x_center, y_center, width, height = box
        left = max(int(x_center - width / 2), 0)
        top = max(int(y_center - height / 2), 0)
        right = min(int(x_center + width / 2), 640)
        bottom = min(int(y_center + height / 2), 640)
        
        face_image = frame_resized[top:bottom, left:right]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(face_image_rgb)

        name = "Unknown"
        if face_encodings:
            face_encoding = face_encodings[0]
            face_encoding_gpu = torch.tensor(face_encoding).to(device)
            
            face_distances = torch.linalg.norm(known_face_encodings_gpu - face_encoding_gpu, dim=1)
            best_match_index = torch.argmin(face_distances).item()
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]

        cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame_resized, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_resized, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return frame_resized

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Xử lý frame
    processed_frame = process_frame(frame)

    # Hiển thị frame
    cv2.imshow('Real-time Detection and Recognition', processed_frame)

    # Xử lý phím nhấn
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()