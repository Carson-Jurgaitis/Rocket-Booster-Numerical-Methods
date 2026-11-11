import numpy as np
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

#defining constants
A = 38
Cd = 1.12
G = 6.674 * 10**-11
ME = 5.972 * 10**24
md = 1.3*10**5
zE = 6.371*10**6

#implementing Equation (2.19)
def rho(x):
    if x > 2.5 * 10 ** 4:
        return 0.040015 * ((141.94+2.99*x*10**-3)/216.6)**-12.388
    elif 1.1 * 10**4 < x < 2.5 * 10 ** 4:
        return 0.3641*np.exp(1.73-1.57*x*10**-4)
    else:
        return 1.225 * ((288.19-6.49*x*10**-3)/288.08)**4.256

#implementing Equation (2.28)
def zddot(z,zdot):
    return Cd*rho(z)*A*zdot**2/(2*md)-G*ME/(z+zE)**2

#making the time array
t_array = np.linspace(0,100,100000)

#writing initial conditions
z_initial = 7.0*10**4
zdot_initial = -1.6667*10**3

#solving the differential equation
def RK4(t,z_i,zdot_i):
    #set initial conditions
    z = np.zeros(len(t))
    z[0] = z_i
    zdot = np.zeros(len(t))
    zdot[0] = zdot_i
    h = t[1]-t[0]
    #running the RK4 method
    for i in range(1,len(t)):
        if z[i-1] > 0:
            k1 = h*zdot[i-1]
            l1 = h*zddot(z[i-1],zdot[i-1])
            k2 = h*(zdot[i-1]+l1/2)
            l2 = h*zddot(z[i-1]+k1/2,zdot[i-1]+l1/2)
            k3 = h*(zdot[i-1]+l2/2)
            l3 = h*zddot(z[i-1]+k2/2,zdot[i-1]+l2/2)
            k4 = h*(zdot[i-1]+l3)
            l4 = h*zddot(z[i-1]+k3,zdot[i-1]+l3)

            z[i] = z[i-1] + 1/6 * (k1+2*k2+2*k3+k4)
            zdot[i] = zdot[i-1] + 1/6 * (l1+2*l2+2*l3+l4)
            
        else:
            #once we reach the ground, z should stay at zero and zdot should also be zero
            #note the time we hit the ground as t_b
            if z[i-2] > 0 and z[i-1] <= 0:
                t_f = t[i-1]
            z[i] = 0
            zdot[i] = 0
            

    return z, zdot, t_f

#get t_bmax from the RK4 function
t_bmax = RK4(t_array,z_i=70000,zdot_i=-1666.7)[2]

#plot the position vs time
plt.subplot(211)
plt.plot(t_array,RK4(t_array,z_initial,zdot_initial)[0],'r')
plt.axvline(x=t_bmax, color='k',linestyle=":", label=r"$t_f$")
plt.legend()
plt.title("Position vs. Time of the Booster")
plt.ylabel(r'$z$ (m)')
plt.xlabel('time (s)')

#plot the velocity vs time
plt.subplot(212)
plt.plot(t_array,RK4(t_array,z_i=70000,zdot_i=-1666.7)[1],'b')
plt.axvline(x=t_bmax, color='k',linestyle=":", label=r"$t_f$")
plt.legend()
plt.title("Velocity vs. Time of the Booster")
plt.ylabel(r'$\dot{z}$ (m/s)')
plt.xlabel('time (s)')

#print t_b and display the plots
print(t_bmax)
plt.tight_layout()
plt.show()
