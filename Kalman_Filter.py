import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

class Simulated_Kalman_Filter:
    def __init__(self, duration, num_samples, drift, Q, R):
        # Time info
        self.t = np.linspace(0, duration, num_samples)
        self.dt = float(duration)/num_samples

        # Generate true signal and noisy signals
        self.test_signal = np.cos(self.t)
        self.measured_signal = self.test_signal + np.random.normal(0,0.1,num_samples)
        self.measured_rate = np.gradient(self.test_signal) + drift*self.t + np.random.normal(0,0.1,num_samples)

        # Initialize internal state representation
        self.K = np.matrix([[1.0], [1.0]])
        self.H = np.matrix([1, 0])
        self.P = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        self.Q = Q
        self.R = R
        self.drift = drift

        # Initialize plotting elements
        self.fig = plt.figure()
        self.plot, = plt.plot([], [])
        self.ax = self.fig.add_subplot(111)

    def update_graph(self, x, y):
        self.plot.set_xdata(np.append(self.plot.get_xdata(), x))
        self.plot.set_ydata(np.append(self.plot.get_ydata(), y))
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        
    def Simulate(self):
        plt.ion()
        plt.show()
        plt.plot(self.t, self.measured_signal)
        plt.plot(self.t, self.test_signal)
        angle = [0]
        bias = [0]
        for i in range(1, len(self.t)):
            angle.append(angle[i-1] + (self.measured_rate[i] - bias[i-1])*self.dt)

            self.P[0, 0] += self.dt*(self.dt*self.P[1, 1] - self.P[0, 1] - self.P[1, 0] + self.Q)
            self.P[0, 1] -= self.dt*self.P[1, 1]
            self.P[1, 0] -= self.dt*self.P[1, 1]
            self.P[1, 1] += self.drift*self.dt

            K1 = float(self.P[0, 0]) / (self.P[0, 0]+self.R)
            K2 = float(self.P[1, 0]) / (self.P[0, 0]+self.R)
            self.K[0, 0] = K1
            self.K[1, 0] = K2

            angle[i] = angle[i] + self.K[0, 0]*(self.measured_signal[i] - angle[i])
            bias.append(bias[i-1] + self.K[1, 0]*(self.measured_signal[i] - angle[i]))

            p_update = (1-self.K[0, 0] + self.K[1, 0])
            self.P[0, 0] = p_update*self.P[0, 0]
            self.P[0, 1] = p_update*self.P[0, 1]
            self.P[1, 0] = p_update*self.P[1, 0]
            self.P[1, 1] = p_update*self.P[1, 1]

            self.update_graph(i*self.dt, angle[i])

        while True:
            pass

sim = Simulated_Kalman_Filter(30, 500, 0.2, 0.01, 0.01)
sim.Simulate()
