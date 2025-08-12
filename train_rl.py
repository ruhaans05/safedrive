import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import cv2 # OpenCV for image processing

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# --- Configuration ---
SECONDS_PER_EPISODE = 45
IMG_WIDTH = 224
IMG_HEIGHT = 224

# --- The CARLA Environment Wrapper ---
class CarlaEnv(gym.Env):
    """
    The CARLA environment wrapper that follows the gymnasium.Env interface.
    This is the bridge between the CARLA simulator and the RL agent.
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
        self.prev_steer = 0.0

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        Destroys old actors and spawns a new vehicle.
        """
        self._cleanup()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Try to find a valid spawn point
        spawn_point = None
        while spawn_point is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else None
            if spawn_point is None:
                print("Couldn't find spawn points, retrying...")
                time.sleep(1)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # Attach sensors
        # 1. RGB Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda data: self._process_img(data))
        self.actor_list.append(self.camera_sensor)

        # 2. Collision Sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)
        self.collision_hist = []
        
        # Wait for the first image to be captured
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start_time = time.time()
        self.prev_steer = 0.0
        return self.front_camera, {}

    def step(self, action):
        """
        Executes one step in the environment.
        Applies the action, gets the new observation, and calculates the reward.
        """
        steer, throttle, brake = action
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        self.vehicle.apply_control(control)

        # Wait for a small amount of time to let the simulation progress
        time.sleep(0.1) 
        reward, terminated = self._calculate_reward()

        if time.time() > self.episode_start_time + SECONDS_PER_EPISODE:
            terminated = True # End episode if it runs too long

        # Update previous steer for next step's calculation
        self.prev_steer = float(steer)

        return self.front_camera, reward, terminated, False, {}

    def _calculate_reward(self):
        """
        Defines the enhanced reward function.
        """
        # --- Positive Rewards ---
        # 1. Reward for speed
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_reward = min(speed_kmh / 10.0, 3.0) # Reward up to 30km/h

        # 2. Reward for staying in lane
        waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        vehicle_transform = self.vehicle.get_transform()
        vector_to_waypoint = vehicle_transform.location - waypoint.transform.location
        distance_from_center = np.sqrt(vector_to_waypoint.x**2 + vector_to_waypoint.y**2)
        lane_reward = max(0.0, 1.0 - (distance_from_center / (waypoint.lane_width / 2.0)))
        
        # --- Negative Rewards (Penalties) ---
        # 1. Penalty for jerky steering
        steer_diff = abs(self.vehicle.get_control().steer - self.prev_steer)
        smoothness_penalty = -steer_diff * 10.0
        
        # 2. Collision penalty
        terminated = False
        collision_penalty = 0.0
        if len(self.collision_hist) > 0:
            terminated = True
            collision_penalty = -100.0 # Heavy penalty

        total_reward = speed_reward + lane_reward + smoothness_penalty + collision_penalty
        return total_reward, terminated

    def _on_collision(self, event):
        self.collision_hist.append(event)

    def _process_img(self, image):
        i = np.array(image.raw_data).reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
        self.front_camera = i

    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.front_camera = None

    def close(self):
        self._cleanup()

def main():
    # --- Create and Check the Environment ---
    print("Creating CARLA environment...")
    env = CarlaEnv()
    check_env(env)
    print("Environment created successfully!")

    # --- Define and Train the RL Model ---
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=1, 
        tensorboard_log="./safedrive_ppo_tensorboard/"
    )

    print("Starting RL training...")
    # Train for a total of 50,000 steps. For real results, this number should be much higher.
    model.learn(total_timesteps=50000)

    # --- Save the Trained Model ---
    model.save("safedrive_ppo_model")
    print("Training complete. Model saved as safedrive_ppo_model.zip")

    # --- Clean up ---
    env.close()

if __name__ == "__main__":
    main()
