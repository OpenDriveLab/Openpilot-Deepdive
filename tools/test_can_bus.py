from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import numpy as np
import math


sensors_tree = {
    'ms_imu':
    [
        'linear_accel',
        'q',
        'rotation_rate',
    ],

    'pose':
    [
        'accel',
        'orientation',
        'pos',
        'rotation_rate',
        'vel',
    ],

    'steeranglefeedback':
    [
        'value',
    ],

    'vehicle_monitor':
    [
        'available_distance',
        'battery_level',
        'brake',
        'brake_switch',
        'gear_position',
        'left_signal',
        'rear_left_rpm',
        'rear_right_rpm',
        'right_signal',
        'steering',
        'steering_speed',
        'throttle',
        'vehicle_speed',
        'yaw_rate',
    ],

    'zoe_veh_info':
    [
        'FL_wheel_speed',
        'FR_wheel_speed',
        'RL_wheel_speed',
        'RR_wheel_speed',
        'left_solar',
        'longitudinal_accel',
        'meanEffTorque',
        'odom',
        'odom_speed',
        'pedal_cc',
        'regen',
        'requestedTorqueAfterProc',
        'right_solar',
        'steer_corrected',
        'steer_offset_can',
        'steer_raw',
        'transversal_accel',
    ],

    'zoesensors':
    [
        'brake_sensor',
        'steering_sensor',
        'throttle_sensor',
    ],
}


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


nusc_can = NuScenesCanBus(dataroot='/data/nuscenes_all')

scene_name = 'scene-0001'
nusc_can.print_all_message_stats(scene_name)

# vis - check
for message_name, keys in sensors_tree.items():
    for key_name in keys:
        nusc_can.plot_message_data(scene_name, message_name, key_name)

# Retrieve raw data.
wheel_speed = nusc_can.get_messages(scene_name, 'zoe_veh_info')
wheel_speed = np.array([(m['utime'], m['FL_wheel_speed']) for m in wheel_speed])

