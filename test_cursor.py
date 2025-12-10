from utils.config import visualize_config
import cv2
out = visualize_config('config/demo_config/road_config/background_simple_road.json')
cv2.imwrite('mask_overlay.jpg', out)
print('saved mask_overlay.jpg')