class Kalman_Filter:
    # Initialization function
    def __init__(self, Q_angle=0.0, Q_bias=0.0, R=0.0):
        self.K = [1.0, 1.0]
        self.H = [1, 0]
        self.P = [[0, 0],[0, 0]]
        self.Q_angle = Q_angle
        self.Q_bias = Q_bias
        self.R = R
        self.measured_signal = []
        self.measured_rate = []
        self.estimated_signal = []
        self.bias = []

    # Function to set Q for angle
    def set_Q_angle(self, Q_angle):
        self.Q_angle = Q_angle

    # Function to set Q for bias
    def set_Q_bias(self, Q_bias):
        self.Q_bias = Q_bias

    # Function to set R
    def set_R(self, R):
        self.R = R

    # Function to set entire measured signal
    def set_measured_signal(self, measured_signal):
        self.measured_signal = measured_signal

    # Function to add measured signal value
    def add_measured_data(self, measured_data):
        self.measured_signal.append(measured_data)

    # Function to set entire measured rate
    def set_measured_rate(self, measured_rate):
        self.measured_rate = measured_rate

    # Function to add measured signal rate
    def add_measured_rate(self, measured_rate):
        self.measured_rate.append(measured_rate)

    def set_true_signal(self, true_signal):
        self.true_signal = true_signal

    # Function to estimate angle based on previous angle, measured rate and bias
    def estimate_angle(self, prev_angle, measured_rate, bias, dt):
        return (prev_angle + (measured_rate - bias)*dt)

    # Function to estimate priori
    def estimate_priori(self, dt):
        self.P[0][0] += dt*(dt*self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt*self.P[1][1]
        self.P[1][0] -= dt*self.P[1][1]
        self.P[1][1] += self.Q_bias*dt

    # Function to calculate Kalman Gain
    def calculate_gain(self):
        self.K[0] = self.P[0][0] / (self.P[0][0] + self.R)
        self.K[1] = self.P[1][0] / (self.P[0][0] + self.R)

    # Calculate angle based on predicted and measured
    def calculate_angle(self, angle, measured_angle):
        return (angle + self.K[0]*(measured_angle - angle))

    # Calculate bias based on prior bias, predicted angle, and measured angle
    def calculate_bias(self, bias, calculated_angle, measured_angle):
        return (bias + self.K[1]*(measured_angle - calculated_angle))

    # Function to update priori
    def update_priori(self):
        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]
        self.P[0][0] -= self.K[0]*P00_temp
        self.P[0][1] -= self.K[0]*P01_temp
        self.P[1][0] -= self.K[1]*P00_temp
        self.P[1][1] -= self.K[1]*P01_temp

    # Main measuring function
    def measure(self, prev_angle, prev_bias, measured_rate, dt):
        est_angle = self.estimate_angle(prev_angle, measured_rate, prev_bias,dt)
        self.estimate_priori(dt)
        self.calculate_gain()
        return est_angle

    # Main updating function
    def update(self, est_angle, measured_angle, prev_bias):
        self.update_priori()
        calculated_angle = self.calculate_angle(est_angle, measured_angle)
        calculated_bias = self.calculate_bias(prev_bias, calculated_angle, measured_angle)
        return (calculated_angle, calculated_bias)

    def estimate_signal(self, dt, initial_angle=0, initial_bias=0):
        estimated_signal = [initial_angle]
        bias = [initial_bias]

        for index in range(1,len(self.measured_signal)):
            estimated_angle = self.measure(estimated_signal[index-1], bias[index-1], self.measured_rate[index], dt[index])
            calculated_angle, calculated_bias = self.update(estimated_angle, self.measured_signal[index], bias[index-1])
            estimated_signal.append(calculated_angle)
            bias.append(calculated_bias)

        return estimated_signal
