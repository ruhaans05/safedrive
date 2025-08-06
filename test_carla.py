# test_carla.py
import carla
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
print("Connected to CARLA:", client.get_world().get_map().name)
