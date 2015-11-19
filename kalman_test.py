import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Sensor_Fusion import Kalman_Filter

Q_angle = 0.001
Q_bias = 0.003
R_measure = 0.03

t = np.arange(0,20,0.01)
dt = [0.01]*len(t)
true_signal = [x**2 for x in t]
measured_signal = true_signal + np.random.normal(0, 1 , len(t))
measured_rate = np.gradient(true_signal) + Q_bias*t + np.random.normal(0, 0.1, len(t))

sensor_fusion = Kalman_Filter(Q_angle, Q_bias, R_measure)
sensor_fusion.set_true_signal(true_signal)
sensor_fusion.set_measured_signal(measured_signal)
sensor_fusion.set_measured_rate(measured_rate)

estimated_signal = sensor_fusion.estimate_signal(dt)

fig, ax = plt.subplots()
line, = ax.plot(t, estimated_signal)
ax.plot(t, measured_signal)
ax.plot(t, true_signal)
plt.show()
