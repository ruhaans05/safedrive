# safety.py
import numpy as np
import carla

def get_pedestrian_distance(world, vehicle):
    """
    Returns distance to closest pedestrian using CARLA ground truth
    """
    vloc = vehicle.get_location()
    walkers = world.get_actors().filter('walker.pedestrian.*')
    
    distances = []
    for walker in walkers:
        wloc = walker.get_location()
        distances.append(vloc.distance(wloc))
    
    return min(distances) if distances else float('inf')



def compute_safety_action(distance):
    """
    Map distance to emergency action
    """
    if distance > 20.0:
        return None  # No override
    elif distance > 10.0:
        return [0.0, 0.0, 0.4]   # Brake moderately
    elif distance > 5.0:
        return [0.0, 0.0, 0.8]   # Hard brake
    else:
        return [-0.5, 0.0, 1.0]  # Swerve left + full stop