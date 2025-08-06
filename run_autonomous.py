# run_autonomous.py
import carla
import torch
import numpy as np
import cv2
from model import SafeDriveNet
from utils import get_transform
import time

# Load model
model = SafeDriveNet(num_actions=3, seq_len=5)
model.load_state_dict(torch.load('safedrive_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Front camera
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '224')
cam_bp.set_attribute('image_size_y', '224')
cam_bp.set_attribute('fov', '110')
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.8)), attach_to=vehicle)

# Transform for preprocessing
transform = get_transform()

# Frame buffer
frame_buffer = []

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    tensor = transform(array).unsqueeze(0).to(device)  # Add batch dim
    return tensor

def camera_callback(image):
    global frame_buffer
    frame_tensor = process_image(image)
    frame_buffer.append(frame_tensor)
    if len(frame_buffer) > 5:
        frame_buffer.pop(0)

camera.listen(camera_callback)

try:
    time.sleep(2)  # Camera warm-up
    print("AI is now driving...")
    while True:
        if len(frame_buffer) == 5:
            with torch.no_grad():
                inputs = torch.cat(frame_buffer, dim=1)  # (1, 5, 3, 224, 224)
                action = model(inputs).cpu().numpy()[0]
            steer, throttle, brake = action[0], action[1], action[2]
            vehicle.apply_control(carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake)
            ))
        time.sleep(0.05)  # ~20 FPS
except KeyboardInterrupt:
    print("\n🛑 AI driving stopped.")

finally:
    camera.stop()
    vehicle.destroy()