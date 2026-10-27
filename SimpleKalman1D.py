class SimpleKalman1D:

    def __init__(self, q_process_noise=1e-3, r_measurement_noise=1e-1):
        # q_process_noise - how quick will the algotrithm adapt to changes in the signal (higher = faster)
        # r_measurement_noise - how much noise is expected in the signal (higher = more noise)
        
        self.Q = q_process_noise
        self.R = r_measurement_noise
        self.x = None  # initial state
        self.P = 1.0  # initial uncertainty
    
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        
        # Prediction step
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x += K * (measurement - self.x)
        self.P = P_pred * (1 - K)
        
        return self.x