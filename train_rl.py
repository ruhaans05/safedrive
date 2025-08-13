import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# --- Configuration ---
SECONDS_PER_EPISODE = 60
IMG_WIDTH = 224
IMG_HEIGHT = 224

# --- The CARLA Environment Wrapper ---
class CarlaEnv(gym.Env):
    """
    The CARLA environment wrapper with a more sophisticated reward function.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Action and Observation Spaces
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        self.actor_list = []
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.front_camera = None
        
        # <<< NEW: Track previous actions for smoothness calculation >>>
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

    def reset(self, seed=None, options=None):
        self._cleanup()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # Attach sensors...
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IMG_WIDTH}'); camera_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}'); camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda data: self._process_img(data))
        self.actor_list.append(self.camera_sensor)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)
        self.collision_hist = []
        
        while self.front_camera is None: time.sleep(0.01)
        self.episode_start_time = time.time()
        
        # Reset previous action tracking
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

        return self.front_camera, {}

    def step(self, action):
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        self.vehicle.apply_control(control)

        time.sleep(0.1) 
        reward, terminated = self._calculate_reward()

        if time.time() > self.episode_start_time + SECONDS_PER_EPISODE:
            terminated = True

        # Update previous actions for the next step's calculation
        self.prev_steer = steer
        self.prev_throttle = throttle
        self.prev_brake = brake

        return self.front_camera, reward, terminated, False, {}

    def _calculate_reward(self):
        """
        Defines the enhanced, more nuanced reward function.
        """
        # --- Positive Rewards ---
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_reward = min(speed_kmh / 10.0, 3.0)

        # <<< NEW: Comfort Bonus for low acceleration >>>
        accel = self.vehicle.get_acceleration()
        accel_magnitude = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        comfort_bonus = max(0.0, 1.0 - (accel_magnitude / 10.0)) # Reward for accel < 10 m/s^2

        # --- Negative Rewards (Penalties) ---
        # <<< NEW: Continuous Lane Deviation Penalty >>>
        waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        distance_from_center = np.sqrt((self.vehicle.get_location().x - waypoint.transform.location.x)**2 + (self.vehicle.get_location().y - waypoint.transform.location.y)**2)
        lane_deviation_penalty = - (distance_from_center / (waypoint.lane_width / 2.0)) * 2.0

        # <<< NEW: Penalties for jerky throttle and brake >>>
        control = self.vehicle.get_control()
        steer_diff = abs(control.steer - self.prev_steer)
        throttle_diff = abs(control.throttle - self.prev_throttle)
        brake_diff = abs(control.brake - self.prev_brake)
        smoothness_penalty = -(steer_diff * 5.0 + throttle_diff * 2.0 + brake_diff * 2.0)
        
        # Collision penalty
        terminated = False
        collision_penalty = 0.0
        if len(self.collision_hist) > 0:
            terminated = True
            collision_penalty = -200.0 # Increased penalty

        # --- Total Reward ---
        total_reward = (
            speed_reward + 
            comfort_bonus +
            lane_deviation_penalty + 
            smoothness_penalty + 
            collision_penalty
        )
        
        return total_reward, terminated

    def _on_collision(self, event): self.collision_hist.append(event)
    def _process_img(self, image):
        i = np.array(image.raw_data).reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
        self.front_camera = i
    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive: actor.destroy()
        self.actor_list = []
    def close(self): self._cleanup()

def main():
    env = CarlaEnv()
    check_env(env)
    
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./safedrive_ppo_tensorboard/")
    
    print("Starting RL training with refined reward function...")
    model.learn(total_timesteps=100000) # Increased timesteps for more complex learning
    
    model.save("safedrive_ppo_model_refined")
    print("Training complete. Model saved as safedrive_ppo_model_refined.zip")
    
    env.close()

if __name__ == "__main__":
    main()
