from ultralytics import YOLO



def main():
    model = YOLO("yolov10n.pt")
    model.train(data=r"D:\pythonlearn\Person-Faces-5\data.yaml", epochs=1000, imgsz=640)

if __name__ == '__main__':
    main()