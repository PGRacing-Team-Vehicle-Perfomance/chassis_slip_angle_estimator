import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


class SteeringKinematics:
    def __init__(self, calibration_csv='steering_system.csv'):
        lut = pd.read_csv(calibration_csv).sort_values('swa_deg')

        # changing the degreed to radians
        swa_rad = np.radians(lut['swa_deg'].values)
        wl_rad = np.radians(lut['wheel_left_deg'].values)
        wr_rad = np.radians(lut['wheel_right_deg'].values)

        self.interp_left = PchipInterpolator(swa_rad, wl_rad, extrapolate=True)
        self.interp_right = PchipInterpolator(swa_rad, wr_rad, extrapolate=True)

    def get_wheel_angles(self, swa_deg):
        swa_rad = np.radians(swa_deg)
        delta_left = float(self.interp_left(swa_rad))
        delta_right = float(self.interp_right(swa_rad))
        return delta_left, delta_right
