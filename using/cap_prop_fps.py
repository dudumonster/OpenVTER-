import cv2

path = r"data1\UAV_Videos\20220303_5_E_300\track\test_multiclass.mp4"
cap = cv2.VideoCapture(path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Resolution:", f"{w}x{h}")
print("FPS:", fps)
print("Frames:", frames)

cap.release()
