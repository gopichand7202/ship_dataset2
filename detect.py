import cv2
import torch
from ultralytics import YOLO
import PySpin  # For FLIR camera support

class FLIRCamera:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list[0] if self.cam_list.GetSize() > 0 else None

        if self.cam is None:
            raise RuntimeError("No FLIR camera detected.")

        self.cam.Init()

    def read_frame(self):
        image_result = self.cam.GetNextImage()
        if image_result.IsIncomplete():
            print("Image incomplete with status %d ..." % image_result.GetImageStatus())
            return None

        frame = image_result.GetNDArray()
        image_result.Release()
        return frame

    def release(self):
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()

# Load YOLOv5 model
model = YOLO('yolov5s.pt')

# Initialize FLIR camera
flir_cam = FLIRCamera()

try:
    while True:
        frame = flir_cam.read_frame()
        if frame is not None:
            # Convert FLIR frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)

            # Perform YOLOv5 object detection
            results = model.predict(source=rgb_frame)

            # Display results
            cv2.imshow('FLIR Camera Detection', results[0].plot())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    flir_cam.release()
    cv2.destroyAllWindows()
