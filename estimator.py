import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from SimpleKalman1D import SimpleKalman1D
from KinematicKalmanFilter import KinematicKalmanFilter


# All calculations are based on article
# "Kinematics-Based Analytical Solution for Wheel Slip
# Angle Estimation of a RWD Vehicle with Drift by Ronnapee Chaichaowarata
# and Witaya Wannasuphoprasiy

# CONSTANT VEHICLE PARAMETERS
# right now, these are hardcoded, but they should be read from a config file
# or passed as arguments
MASS_KG = 1000.0
WHEELBASE_M = 2.5
TRACK_WIDTH_M = 1.5
L_F = 1.2  # CG_TO_FRONT_AXLE_M
L_R = 1.3  # CG_TO_REAR_AXLE_M
DT = 0.01  # time step between the samples in seconds (0.01s = 100Hz)
STEERING_RATIO = 15.0  # steering wheel angle to wheel angle ratio

# Axis conventions:
# X is positive forward, Y is positive to the right, Z is positive up
# Yaw rate is positive counterclockwise
# Wheel angle, chassis slip angle and wheel slip angles
# are positive to the right (clockwise)

samples = pd.read_csv('telemetry.csv')

data_for_estimation = samples[['front_right_wheel_speed',
                               'front_left_wheel_speed',
                               'yaw_rate', 'imu_acceleration_y',
                               'imu_acceleration_x',
                               'steering_wheel_angle']]


def estimate_side_kinematics(sample, wheel, eps=1e-5):
    Vrx = sample['front_right_wheel_speed']
    Vlx = sample['front_left_wheel_speed']
    yaw_rate = sample['yaw_rate']
    a_x = sample['imu_acceleration_x']
    a_y = sample['imu_acceleration_y']
    # TODO: write a function for converting SWA to wheel angle,
    # in case we don't have STEERING_RATIO
    delta = np.radians(sample['steering_wheel_angle']) / STEERING_RATIO

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    if wheel.lower() == "right":
        sign = 1
        V = Vrx
    else:
        sign = -1
        V = Vlx

    w_eff = sign * TRACK_WIDTH_M

    A = (V**2) * yaw_rate
    B = V * (
      (yaw_rate**2) * (2.0 * L_F * cos_d + w_eff * sin_d)
      - (a_x * cos_d + a_y * sin_d)
    )
    C = (
        V
        * (
            (yaw_rate**2) * (2.0 * L_F * sin_d - w_eff * cos_d)
            - (a_x * sin_d - a_y * cos_d)
        )
        + (L_F**2 + (w_eff**2) / 4.0) * (yaw_rate**3)
        + (V**2 - a_x * L_F - a_y * (w_eff / 2.0)) * yaw_rate
    )

    if abs(A) < eps:
        tan_alpha_f = -C / B if abs(B) > eps else 0.0
    else:
        discriminant = max(0.0, B**2 - 4.0 * A * C)
        sign_B = 1.0 if B >= 0.0 else -1.0
        tan_alpha_f = (-B + sign_B * math.sqrt(discriminant)) / (2.0 * A)

    alpha_f = math.atan(tan_alpha_f)

    return alpha_f  # in radians!!


# Initialization of Kalman filters for inputs
# (wheel speeds, yaw rate, accelerations)
kf_vr = SimpleKalman1D(q_process_noise=1e-2, r_measurement_noise=1e-1)
kf_vl = SimpleKalman1D(q_process_noise=1e-2, r_measurement_noise=1e-1)
kf_yaw = SimpleKalman1D(q_process_noise=1e-3, r_measurement_noise=2e-1)
kf_ax = SimpleKalman1D(q_process_noise=1e-2, r_measurement_noise=5e-1)
kf_ay = SimpleKalman1D(q_process_noise=1e-2, r_measurement_noise=5e-1)

# Initialization of Kalman filters for outputs (sideslip angles)
kf_alpha_fr = SimpleKalman1D(q_process_noise=1e-3, r_measurement_noise=1e-1)
kf_alpha_fl = SimpleKalman1D(q_process_noise=1e-3, r_measurement_noise=1e-1)
kf_beta = KinematicKalmanFilter(dt=DT, q_noise=1e-4, r_noise=1e-1)

alpha_fr_list = []
alpha_fl_list = []
beta_list = []

for i in range(len(samples)):
    raw_sample = samples.iloc[i]

    # Filtering the raw telemetry data using simple Kalman filter
    filtered_sample = {
        'front_right_wheel_speed': kf_vr.update(
         raw_sample['front_right_wheel_speed']
        ),
        'front_left_wheel_speed': kf_vl.update(
            raw_sample['front_left_wheel_speed']
        ),
        'yaw_rate': kf_yaw.update(raw_sample['yaw_rate']),
        'imu_acceleration_x': kf_ax.update(raw_sample['imu_acceleration_x']),
        'imu_acceleration_y': kf_ay.update(raw_sample['imu_acceleration_y']),
        'steering_wheel_angle': raw_sample['steering_wheel_angle'],
    }

    Vfx = filtered_sample['front_right_wheel_speed']
    steering_angle = np.radians(
        filtered_sample['steering_wheel_angle']) / STEERING_RATIO
    yaw_rate = filtered_sample['yaw_rate']
    a_y = filtered_sample['imu_acceleration_y']

    # FRONT RIGHT WHEEL SIDESLIP ANGLE (ALPHA_FR) ESTIMATION
    alpha_fr_raw = estimate_side_kinematics(filtered_sample, "right")
    alpha_fr = kf_alpha_fr.update(alpha_fr_raw)

    # FRONT LEFT WHEEL SIDESLIP ANGLE (ALPHA_FL) ESTIMATION
    alpha_fl_raw = estimate_side_kinematics(filtered_sample, "left")
    alpha_fl = kf_alpha_fl.update(alpha_fl_raw)

    alpha_fr_list.append(alpha_fr)
    alpha_fl_list.append(alpha_fl)

    # Estimation of Vx and Vy of the vehicle
    Vx = ((Vfx / np.cos(alpha_fr)) * np.cos(steering_angle + alpha_fr)) - (
        TRACK_WIDTH_M/2.0 * yaw_rate)
    Vy = ((Vfx / np.cos(alpha_fr)) * np.sin(steering_angle + alpha_fr)) + (
        L_F * yaw_rate)

    # Chassis slip angle estimation based on the estimated Vx and Vy
    beta_raw = np.arctan2(Vy, Vx)
    V_total = np.hypot(Vx, Vy)

    # Chassis slip angle estimation using the kinematic Kalman filter
    beta = kf_beta.step(beta_raw, a_y, yaw_rate, V_total)

    beta_list.append(beta)


samples['alpha_fr_rad'] = alpha_fr_list
samples['alpha_fl_rad'] = alpha_fl_list
samples['beta_rad'] = beta_list


# Printing the plot of the estimated angles in time
plt.figure(figsize=(12, 6))
plt.plot(samples['alpha_fr_rad'], label='FR Wheel Sideslip Angle (rad)')
plt.plot(samples['alpha_fl_rad'], label='FL Wheel Sideslip Angle (rad)')
plt.plot(samples['beta_rad'], label='Chassis Slip Angle (rad)')
plt.xlabel('Time')
plt.ylabel('Angle (rad)')
plt.title('Estimated Angles Over Time')
plt.legend()
plt.show()
