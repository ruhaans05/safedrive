# collect_data.py
import carla
import numpy as np
import cv2
import os
import time

# --- Setup ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
spawn = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn)

# --- Camera Blueprint (RGB) ---
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '224')
cam_bp.set_attribute('image_size_y', '224')
cam_bp.set_attribute('fov', '110')

# --- Camera Blueprint (Depth) ---
depth_bp = bp_lib.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '224')
depth_bp.set_attribute('image_size_y', '224')
depth_bp.set_attribute('fov', '110')

# --- Camera Transforms ---
def at_front():
    return carla.Transform(carla.Location(x=1.5, z=1.8))

def at_left():
    return carla.Transform(carla.Location(x=0, y=-0.5, z=1.8), carla.Rotation(yaw=-60))

def at_right():
    return carla.Transform(carla.Location(x=0, y=0.5, z=1.8), carla.Rotation(yaw=60))

def at_rear():
    return carla.Transform(carla.Location(x=-1.5, z=1.8), carla.Rotation(yaw=180))

# --- Spawn Cameras ---
front_cam = world.spawn_actor(cam_bp, at_front(), attach_to=vehicle)
left_cam  = world.spawn_actor(cam_bp, at_left(),  attach_to=vehicle)
right_cam = world.spawn_actor(cam_bp, at_right(), attach_to=vehicle)
rear_cam  = world.spawn_actor(cam_bp, at_rear(),  attach_to=vehicle)
depth_cam = world.spawn_actor(depth_bp, at_front(), attach_to=vehicle)  # Same location as front cam

# --- Create Folders ---
data_root = f"data_{int(time.time())}"
os.makedirs(data_root, exist_ok=True)
os.makedirs(f"{data_root}/front", exist_ok=True)
os.makedirs(f"{data_root}/left", exist_ok=True)
os.makedirs(f"{data_root}/right", exist_ok=True)
os.makedirs(f"{data_root}/rear", exist_ok=True)
os.makedirs(f"{data_root}/depth", exist_ok=True)
os.makedirs(f"{data_root}/controls", exist_ok=True)

# --- Frame Counter ---
frame_id = 0

# --- Save Functions ---
def save_front(image):
    global frame_id
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    cv2.imwrite(f"{data_root}/front/{frame_id:06d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def save_left(image):
    global frame_id
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    cv2.imwrite(f"{data_root}/left/{frame_id:06d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def save_right(image):
    global frame_id
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    cv2.imwrite(f"{data_root}/right/{frame_id:06d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def save_rear(image):
    global frame_id
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    cv2.imwrite(f"{data_root}/rear/{frame_id:06d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def save_depth(image):
    global frame_id
    # CARLA encodes depth as a pseudo-color image
    # Extract R, G, B channels
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    r = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    b = array[:, :, 2].astype(np.float32)
    # Convert to depth in meters
    normalized = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
    depth_meters = 1000 * normalized  # 0-1000m
    # Save as PNG (normalized to 0-255 for visualization)
    depth_vis = (depth_meters / 1000 * 255).astype(np.uint8)  # Normalize to 0-255
    cv2.imwrite(f"{data_root}/depth/{frame_id:06d}.png", depth_vis)

def save_controls():
    global frame_id
    control = vehicle.get_control()
    np.save(f"{data_root}/controls/{frame_id:06d}.npy", {
        'steer': float(control.steer),
        'throttle': float(control.throttle),
        'brake': float(control.brake),
        'reverse': bool(control.reverse),
        'hand_brake': bool(control.hand_brake)
    })

# --- Attach Listeners ---
front_cam.listen(save_front)
left_cam.listen(save_left)
right_cam.listen(save_right)
rear_cam.listen(save_rear)
depth_cam.listen(save_depth)

# --- Start Recording ---
print("🚗 Recording 360° driving data with depth... Drive manually for 30 seconds.")
try:
    while frame_id < 600:  # Record for ~30 seconds at ~20 FPS
        # Save controls on every frame
        if frame_id < 600:  # Only save if we're still recording
            save_controls()
        time.sleep(0.05)  # ~20 FPS
        frame_id += 1
except KeyboardInterrupt:
    print("\n🛑 Recording interrupted by user.")

finally:
    print(f"✅ Recorded {frame_id} frames to {data_root}")
    front_cam.stop()
    left_cam.stop()
    right_cam.stop()
    rear_cam.stop()
    depth_cam.stop()
    vehicle.destroy()