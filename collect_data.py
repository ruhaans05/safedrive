import carla
import numpy as np
import cv2
import os
import time
import argparse
import json

# --- CLI Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
parser.add_argument("--fps", type=int, default=20, help="Frames per second")
parser.add_argument("--name", type=str, default="drive", help="Name of data folder prefix")
args = parser.parse_args()

# --- Setup ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# --- Sensor Blueprints ---
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '224'); cam_bp.set_attribute('image_size_y', '224'); cam_bp.set_attribute('fov', '110')

lidar_bp = bp_lib.find('sensor.lidar.ray_cast') # <<< NEW >>>
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('points_per_second', '100000')

# --- Sensor Transforms ---
front_cam_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
lidar_transform = carla.Transform(carla.Location(z=1.9)) # <<< NEW >>>

# --- Spawn Sensors ---
front_cam = world.spawn_actor(cam_bp, front_cam_transform, attach_to=vehicle)
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle) # <<< NEW >>>
sensors = [front_cam, lidar_sensor]

# --- Create Folders ---
timestamp = int(time.time())
data_root = f"{args.name}_{timestamp}"
os.makedirs(data_root, exist_ok=True)
os.makedirs(f"{data_root}/front", exist_ok=True)
os.makedirs(f"{data_root}/lidar", exist_ok=True) # <<< NEW >>>

# --- Frame & Control Logging ---
frame_id = 0
controls_log = []

def save_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((224, 224, 4))[:, :, :3]
    cv2.imwrite(f"{data_root}/front/{frame_id:06d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def save_lidar(point_cloud): # <<< NEW >>>
    """Processes LiDAR data into a 2D Bird's-Eye View image."""
    points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    
    # Filter points to a 50x50m area in front of the car
    lidar_range = 50.0
    points = points[np.abs(points[:, 0]) < lidar_range]
    points = points[np.abs(points[:, 1]) < lidar_range]

    # Convert to a 2D grid
    bev_image = np.zeros((224, 224), dtype=np.uint8)
    for p in points:
        x, y = p[0], p[1]
        # Convert coordinates to pixel locations
        px = int(112 + x)
        py = int(112 - y)
        if 0 <= px < 224 and 0 <= py < 224:
            bev_image[py, px] = 255 # Mark point as white
            
    cv2.imwrite(f"{data_root}/lidar/{frame_id:06d}.png", bev_image)

def save_controls():
    control = vehicle.get_control()
    velocity = vehicle.get_velocity()
    speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    controls_log.append({
        'frame': frame_id, 'steer': float(control.steer), 'throttle': float(control.throttle),
        'brake': float(control.brake), 'speed_kmh': float(speed_kmh)
    })

# --- Attach Listeners ---
front_cam.listen(save_rgb)
lidar_sensor.listen(save_lidar) # <<< NEW >>>

# --- Start Recording ---
max_frames = args.duration * args.fps
print(f"📹 Recording for {args.duration}s at {args.fps} FPS → {max_frames} frames.")
print(f"💾 Saving data to: {data_root}")

try:
    while frame_id < max_frames:
        save_controls()
        time.sleep(1 / args.fps)
        frame_id += 1
except KeyboardInterrupt:
    print("\n🛑 Recording interrupted manually.")
finally:
    for s in sensors:
        if s.is_listening: s.stop()
    with open(f"{data_root}/controls.json", "w") as f:
        json.dump(controls_log, f, indent=2)
    if vehicle.is_alive: vehicle.destroy()
    print(f"\n✅ Done. {frame_id} frames saved in {data_root}")
