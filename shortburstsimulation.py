import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,3.0,256)
v_T = 300
F_T = 2.5*10 ** 6
m = 2 * 10 ** 5
g = 9.8

def v(time):
    v = np.zeros(256)
    for i in range(0,256,1):
        if time[i] < 0.5:
            v[i] = v_T
        elif 0.5 < time[i] < 1:
            v[i] = v_T - F_T / m * (t[i]-0.5)
        else:
            v[i] = v_T * np.tanh(9.8*(t[i]-1) + np.arctanh((v_T - F_T / m * (0.5) )/ v_T))

    return v

plt.plot(t,v(t),'k')
plt.title("Short Thruster Burst")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.show()
