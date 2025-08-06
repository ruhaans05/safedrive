# pipeline.py
import carla
import math
import numpy as np

def carla_depth_to_meters(depth_image):
    r = depth_image[:, :, 0].astype(np.float32)
    g = depth_image[:, :, 1].astype(np.float32)
    b = depth_image[:, :, 2].astype(np.float32)
    normalized = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
    return 1000 * normalized

def get_traffic_light_state(world, vehicle, max_dist=50.0):
    vloc = vehicle.get_location()
    tls = world.get_actors().filter('traffic.traffic_light.*')
    for tl in tls:
        if vloc.distance(tl.get_location()) < max_dist:
            if is_vehicle_facing(vehicle, tl):
                return tl.get_state(), tl.get_transform().location
    return None, None

def is_vehicle_facing(vehicle, tl):
    vtr = vehicle.get_transform()
    ttr = tl.get_transform()
    heading = vtr.rotation.yaw
    angle = math.degrees(math.atan2(ttr.location.y - vtr.location.y, ttr.location.x - vtr.location.x))
    return abs((heading - angle + 180) % 360 - 180) < 30

def should_stop_at_yellow(vehicle, tl_location, tl_state):
    if tl_state != carla.TrafficLightState.Yellow:
        return False
    vloc = vehicle.get_location()
    distance = vloc.distance(tl_location)
    speed = vehicle.get_velocity().length()
    reaction_dist = speed * 1.0
    braking_dist = (speed ** 2) / (2 * 4.0)
    stop_dist = reaction_dist + braking_dist
    if distance < stop_dist:
        return False
    elif distance > stop_dist + 5:
        return True
    else:
        return True

def can_make_left_turn(world, vehicle):
    waypoint = world.get_map().get_waypoint(vehicle.get_location())
    left_lane = waypoint.get_left_lane()
    return left_lane is not None and left_lane.lane_type == carla.LaneType.Driving

def get_surroundings(world, vehicle, max_distance=20.0):
    vloc = vehicle.get_location()
    vehicles = world.get_actors().filter('vehicle.*')
    left_cars = []
    right_cars = []
    rear_cars = []
    for v in vehicles:
        if v.id == vehicle.id:
            continue
        loc = v.get_location()
        dx = loc.x - vloc.x
        dy = loc.y - vloc.y
        dist = vloc.distance(loc)
        speed = v.get_velocity().length()
        if dist > max_distance:
            continue
        if dy < -1.0 and abs(dx) < 10:
            left_cars.append({'dist': dist, 'speed': speed})
        elif dy > 1.0 and abs(dx) < 10:
            right_cars.append({'dist': dist, 'speed': speed})
        elif dx < 0 and abs(dy) < 3.0:
            rear_cars.append({'dist': dist, 'speed': speed})
    return left_cars, right_cars, rear_cars

def detect_pothole(seg_image):
    pothole_mask = (seg_image == 20)
    if not pothole_mask.any():
        return None
    y, x = pothole_mask.nonzero()
    center_x = x.mean()
    width = x.max() - x.min()
    height = y.max() - y.min()
    area = len(x)
    severity = 'hole' if (area > 300 and width > 50 and height > 30) else 'pothole'
    return {
        'x': center_x,
        'area': area,
        'severity': severity,
        'in_lane': abs(center_x - 112) < 50
    }

def detect_crosswalk(seg_image):
    markings = (seg_image == 6)
    if not markings.any():
        return False
    h, w = markings.shape
    lower_markings = markings[h//2:, :]
    row_sums = lower_markings.sum(axis=1)
    if np.mean(row_sums > w * 0.5) > 0.3:
        return True
    return False

def should_stop_at_line(vehicle, seg_image, tl_state):
    if tl_state not in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
        return False
    markings = (seg_image == 6)
    h, w = markings.shape
    mid_row = markings[h//2, :]
    if mid_row.sum() > w * 0.8:
        return True
    return False

def should_back_up_safely(world, vehicle):
    if vehicle.get_velocity().x < 0:
        _, _, rear_cars = get_surroundings(world, vehicle)
        return len(rear_cars) == 0 or min(c['dist'] for c in rear_cars) > 3.0
    return True

def detect_road_obstruction(seg_image, depth_image=None):
    road_mask = (seg_image == 7)
    markings_mask = (seg_image == 6)
    pothole_mask = (seg_image == 20)
    valid_road = road_mask | markings_mask | pothole_mask
    h, w = seg_image.shape
    lower_third = slice(h * 2 // 3, h)
    lateral_center = slice(w // 4, 3 * w // 4)
    roi_y = lower_third
    roi_x = lateral_center
    labels_in_roi = seg_image[roi_y, roi_x]
    valid_in_roi = valid_road[roi_y, roi_x]
    anomaly_mask = ~valid_in_roi
    if not np.any(anomaly_mask):
        return None
    anomaly_pixels = labels_in_roi[anomaly_mask]
    unique_classes = set(anomaly_pixels)
    obstruction_classes = unique_classes - {0, 1, 255}
    if not obstruction_classes:
        return None
    total_anomaly_pixels = np.sum(anomaly_mask)
    if depth_image is not None:
        depth_meters = carla_depth_to_meters(depth_image)
        depth_in_roi = depth_meters[roi_y, roi_x]
        avg_depth = np.mean(depth_in_roi[anomaly_mask])
        if avg_depth > 20:  # >20m away → ignore
            return None
        road_depth = np.mean(depth_in_roi[valid_in_roi])
        if avg_depth > road_depth * 1.1:
            pass
        else:
            return None
    if total_anomaly_pixels > 300:
        severity = 'high'
    elif total_anomaly_pixels > 100:
        severity = 'medium'
    else:
        return None
    return {
        'class_ids': list(obstruction_classes),
        'size': total_anomaly_pixels,
        'severity': severity,
        'distance_estimate': 'close' if avg_depth < 10 else 'mid'
    }

def plan_action(world, vehicle, seg_image=None, depth_image=None):
    vloc = vehicle.get_location()
    speed = vehicle.get_velocity().length() * 3.6
    speed_limit = vehicle.get_speed_limit()

    # --- PERCEIVE ---
    left_cars, right_cars, rear_cars = get_surroundings(world, vehicle)
    pothole = detect_pothole(seg_image) if seg_image is not None else None
    has_crosswalk = detect_crosswalk(seg_image) if seg_image is not None else False
    must_stop_at_line = should_stop_at_line(vehicle, seg_image, None) if seg_image is not None else False
    obstruction = detect_road_obstruction(seg_image, depth_image) if seg_image is not None else None
    walkers = world.get_actors().filter('walker.pedestrian.*')

    # --- ALWAYS-ON SAFETY HAZARDS (LEVEL 5) ---
    # 1. IMMEDIATE COLLISION THREAT: Pedestrian in path
    for w in walkers:
        wloc = w.get_location()
        if vloc.distance(wloc) < 8.0 and is_in_front_of_vehicle(vehicle, w):
            return {'action': 'stop', 'reason': 'pedestrian_in_path'}

    # 2. LARGE, CLOSE OBSTRUCTION
    if obstruction and obstruction['severity'] == 'high' and obstruction['distance_estimate'] == 'close':
        if not right_cars or min(c['dist'] for c in right_cars) > 8.0:
            return {'action': 'swerve', 'direction': 'right', 'steer': +0.4}
        elif not left_cars or min(c['dist'] for c in left_cars) > 8.0:
            return {'action': 'swerve', 'direction': 'left', 'steer': -0.4}
        else:
            return {'action': 'stop', 'reason': 'large_obstruction_imminent'}

    # --- TRAFFIC CONTROL (LEVEL 4-3) ---
    tl_state, tl_loc = get_traffic_light_state(world, vehicle)

    if tl_state == carla.TrafficLightState.Red:
        return {'action': 'stop', 'reason': 'red_light'}

    elif tl_state == carla.TrafficLightState.Yellow:
        if should_stop_at_yellow(vehicle, tl_loc, tl_state):
            return {'action': 'stop', 'reason': 'yellow_must_stop'}
        else:
            return {'action': 'proceed', 'reason': 'yellow_go_through'}

    elif tl_state == carla.TrafficLightState.Green:
        if can_make_left_turn(world, vehicle):
            return {'action': 'wait_for_clear', 'turn': 'left', 'reason': 'left_turn'}

    if must_stop_at_line and tl_state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
        return {'action': 'stop', 'reason': 'stop_at_line'}

    # --- SECONDARY HAZARDS (LEVEL 3) ---
    if has_crosswalk:
        for w in walkers:
            if vloc.distance(w.get_location()) < 15.0:
                return {'action': 'slow_down', 'brake': 0.4, 'reason': 'pedestrian_near_crosswalk'}

    # --- ROUTINE HAZARDS (LEVEL 2-1) ---
    if pothole and pothole['in_lane']:
        can_swerve_right = not right_cars or (right_cars and min(c['dist'] for c in right_cars) > 8.0)
        can_swerve_left = not left_cars or (left_cars and min(c['dist'] for c in left_cars) > 8.0)
        if pothole['severity'] == 'hole':
            if pothole['x'] < 112 and can_swerve_right:
                return {'action': 'swerve', 'direction': 'right', 'steer': +0.4}
            elif pothole['x'] >= 112 and can_swerve_left:
                return {'action': 'swerve', 'direction': 'left', 'steer': -0.4}
            else:
                return {'action': 'stop', 'reason': 'dangerous_hole_no_escape'}
        else:
            if pothole['x'] < 112 and can_swerve_right:
                return {'action': 'swerve', 'direction': 'right', 'steer': +0.2}
            elif pothole['x'] >= 112 and can_swerve_left:
                return {'action': 'swerve', 'direction': 'left', 'steer': -0.2}
            else:
                speed_mps = vehicle.get_velocity().length()
                if speed_mps > 8:
                    return {'action': 'slow_down', 'brake': 0.5, 'reason': 'small_pothole_no_escape'}
                else:
                    return {'action': 'continue'}

    if obstruction and obstruction['severity'] == 'medium':
        can_swerve_right = not right_cars or min(c['dist'] for c in right_cars) > 8.0
        can_swerve_left = not left_cars or min(c['dist'] for c in left_cars) > 8.0
        if can_swerve_right and obstruction['size'] > 150:
            return {'action': 'swerve', 'direction': 'right', 'steer': +0.2}
        elif can_swerve_left and obstruction['size'] > 150:
            return {'action': 'swerve', 'direction': 'left', 'steer': -0.2}
        else:
            return {'action': 'slow_down', 'brake': 0.4, 'reason': 'medium_obstruction'}

    if speed > speed_limit + 5:
        return {'action': 'slow_down', 'to': speed_limit}

    if vehicle.get_control().reverse and not should_back_up_safely(world, vehicle):
        return {'action': 'stop', 'reason': 'obstacle_behind'}

    return {'action': 'continue', 'speed': speed}

# Helper function to check if an actor is in front of the vehicle
def is_in_front_of_vehicle(vehicle, actor):
    vtr = vehicle.get_transform()
    aloc = actor.get_location()
    vloc = vtr.location
    heading = vtr.rotation.yaw
    angle = math.degrees(math.atan2(aloc.y - vloc.y, aloc.x - vloc.x))
    return abs((heading - angle + 180) % 360 - 180) < 45