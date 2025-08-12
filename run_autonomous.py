import carla
import torch
import numpy as np
import cv2
import time

from model import SafeDriveNet
from utils import get_transform
from safety import get_pedestrian_distance, compute_safety_action # <<< NEW >>>

# --- Load Model ---
print("Loading imitation learning model...")
model = SafeDriveNet(num_actions=3, seq_len=5)
# Make sure you have a trained model file at this path
model.load_state_dict(torch.load('safedrive_model_best.pth')) 
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Model loaded.")

# --- Connect to CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# --- Spawn Vehicle ---
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# --- Front Camera ---
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '224')
cam_bp.set_attribute('image_size_y', '224')
cam_bp.set_attribute('fov', '110')
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.8)), attach_to=vehicle)

transform = get_transform()
frame_buffer = []

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    # Note: The transform in utils.py should be used if your dataset was trained with it.
    # For simplicity here, we'll do basic conversion.
    img = array.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor

def camera_callback(image):
    global frame_buffer
    if len(frame_buffer) < 5:
        frame_buffer.append(process_image(image))

camera.listen(camera_callback)

try:
    print("Camera warming up...")
    while len(frame_buffer) < 5:
        time.sleep(0.1)
    
    print("✅ AI is now driving...")
    while True:
        # --- SAFETY OVERRIDE LOGIC ---
        # 1. Get distance to nearest pedestrian
        dist = get_pedestrian_distance(world, vehicle)
        # 2. Compute a safety action based on that distance
        safety_action = compute_safety_action(dist)

        if safety_action is not None:
            # If the safety module returns an action, USE IT and ignore the AI
            steer, throttle, brake = safety_action
            print(f"🚨 SAFETY OVERRIDE! Dist: {dist:.2f}m. Action: {safety_action}")
        else:
            # If it's safe, get the AI's action
            with torch.no_grad():
                # Pop the oldest frame and add the new one
                frame_buffer.pop(0)
                # This assumes a new frame is constantly being added by the callback
                # A more robust implementation would use a queue.
                
                # Re-check buffer length
                if len(frame_buffer) == 4: # Wait for next frame
                    time.sleep(0.05)
                    continue

                inputs = torch.cat(frame_buffer, dim=1)
                action = model(inputs).cpu().numpy()[0]
            steer, throttle, brake = action[0], action[1], action[2]

        # Apply the chosen control
        vehicle.apply_control(carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        ))
        
        time.sleep(1/20) # Match FPS

except KeyboardInterrupt:
    print("\n🛑 AI driving stopped.")

finally:
    if 'camera' in locals() and camera.is_listening:
        camera.stop()
    if 'vehicle' in locals() and vehicle.is_alive:
        vehicle.destroy()
    print("Cleaned up actors.")

