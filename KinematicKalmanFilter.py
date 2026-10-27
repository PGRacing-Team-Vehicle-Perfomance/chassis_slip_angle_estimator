class KinematicKalmanFilter:

    def __init__(self, dt=0.01, q_noise=1e-4, r_noise=5e-2):
        self.dt = dt
        self.Q = q_noise
        self.R = r_noise
        self.beta = 0.0
        self.P = 1.0

    def step(self, beta_measured, a_y, yaw_rate, V_total):
        # Prediction from vehicle kinematics: beta_dot = (a_y / V) - yaw_rate
        # (According to assumption: yaw_rate positive to the left (CCW), beta
        # and a_y positive to the right (CW),
        # so beta_dot = a_y / V + yaw_rate)
        if V_total > 1.0:
            beta_dot = (a_y / V_total) + yaw_rate
        else:
            beta_dot = 0.0

        beta_pred = self.beta + beta_dot * self.dt
        P_pred = self.P + self.Q

        # Correction with measurement from the analytical model in the paper
        K = P_pred / (P_pred + self.R)
        self.beta = beta_pred + K * (beta_measured - beta_pred)
        self.P = (1.0 - K) * P_pred
        return self.beta
