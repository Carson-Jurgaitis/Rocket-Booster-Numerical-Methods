import numpy as np
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from numpy.polynomial import Polynomial

#defining constants
A = 38
Cd = 1.12
G = 6.674 * 10**-11
ME = 5.972 * 10**24
md = 1.3*10**5
zE = 6.371*10**6
Isp = 3200
mdot_max = 4215

#implementing Equation (2.19)
def rho(x):
    if x > 2.5 * 10 ** 4:
        return 0.040015 * ((141.94+2.99*x*10**-3)/216.6)**-12.388
    elif 1.1 * 10**4 < x < 2.5 * 10 ** 4:
        return 0.3641*np.exp(1.73-1.57*x*10**-4)
    else:
        return 1.225 * ((288.19-6.49*x*10**-3)/288.08)**4.256

#implementing Equation (2.41)
def m0(mdot,tb,tf):
    return md + mdot * (tf-tb) + 0.5 * (mdot_max-mdot) * (tf-tb)

#implementing Equation (2.42)
def m(mdot,t,tf,tb):
    return m0(mdot,tb,tf) -  mdot * (t-tb) - 0.5 * (mdot_max-mdot)/(tf-tb) * (t-tb)**2

#implementing Equation (2.40)
def zddot(z,zdot,t,tb,tf,mdot):
    if t >= tb:
        return (mdot*(Isp+zdot) - zdot/abs(zdot) * 0.5 * Cd*rho(z)*A*(zdot)**2)/m(mdot,t,tf,tb)-G*ME/(z+zE)**2
    else:
        return -zdot/abs(zdot)*Cd*rho(z)*A*zdot**2/(2*m0(mdot,tb,tf))-G*ME/(z+zE)**2



#making the time array
t_array = np.hstack((np.linspace(0,40,10000),np.linspace(40,50,100000)))

#writing initial conditions
z_initial = 7.0*10**4
zdot_initial = -1.6667*10**3

#solving the differential equation
def RK4(t,tb,tf,mdot,z_i,zdot_i):
    scenario = 0
    #set initial conditions
    z = np.zeros(len(t))
    z[0] = z_i
    zdot = np.zeros(len(t))
    zdot[0] = zdot_i
    #running the RK4 method
    for i in range(1,len(t)):
        h = t[i]-t[i-1]
        if z[i-1] > 0:
            k1 = h*zdot[i-1]
            l1 = h*zddot(z[i-1],zdot[i-1],t[i-1],tb,tf,mdot)
            k2 = h*(zdot[i-1]+l1/2)
            l2 = h*zddot(z[i-1]+k1/2,zdot[i-1]+l1/2,t[i-1]+h/2,tb,tf,mdot)
            k3 = h*(zdot[i-1]+l2/2)
            l3 = h*zddot(z[i-1]+k2/2,zdot[i-1]+l2/2,t[i-1]+h/2,tb,tf,mdot)
            k4 = h*(zdot[i-1]+l3)
            l4 = h*zddot(z[i-1]+k3,zdot[i-1]+l3,t[i],tb,tf,mdot)

            z[i] = z[i-1] + 1/6 * (k1+2*k2+2*k3+k4)
            zdot[i] = zdot[i-1] + 1/6 * (l1+2*l2+2*l3+l4)

            if zdot[i] > 0: 
                if z[i] > 0 and scenario == 0:
                    minpos = z[i]
                    scenario = 1
                    break
            
            if z[i] < 0:
                #once we reach the ground, z should stay at zero and zdot should also be zero
                #note the time we hit the ground as t_b
                tf = t[i]
                if zdot[i] < -5:
                    scenario = 2
                    crashvelo = zdot[i]
                else:
                    scenario = 3
                break

    return z, zdot, tf, scenario


# finding tf
def find_tf(t,tb,mdot,z_i,zdot_i):
    guess = [47.3]
    for i in range(100):
        rk4 = RK4(t,tb,guess[i-1],mdot,z_i,zdot_i)
        guess.append(rk4[2])
        if guess[i]-guess[i-1] < 0.001:
            break
        if rk4[3] == 1:
            break
    return guess[-1], rk4

#finding tb
def find_tb(t,mdot,z_i,zdot_i):
    lower_bound = 0
    upper_bound = 47.3
    result = [[],[0,0,0,0]]
    while result[1][3] != 3:
        guess = (lower_bound+upper_bound)/2
        result = find_tf(t,guess,mdot,z_i,zdot_i)
        print(guess)
        if result[1][3] == 1:
            lower_bound = guess
        elif result[1][3] == 2:
            upper_bound = guess
            if guess < 0.01:
                print("not enough thrust to prevent crash landing")
                return guess, result, 4
        else:
            return guess, result, 0

#iterating over mdot
def mostEfficientMass(mdot,t,z_i,zdot_i):
    mass = np.zeros(len(mdot))
    for i in range(len(mdot)):
        result = find_tb(t,mdot[i],z_i,zdot_i)
        mass[i] = m0(mdot[i],result[0],result[1][0])
        if result[2] == 4:
            mass[i] = 250000
    return mass

Mdot = np.linspace(10812,10812,1)
data = find_tb(t_array,10812,z_initial,zdot_initial)
print(data[1][0])

plt.subplot(211)
plt.plot(t_array,data[1][1][0],'r',label="position")
plt.axvline(x=data[0],ls=':',color='g',label=r'$t_b$')
plt.axvline(x=data[1][0],ls='--',color='k',label=r'$t_f$')
plt.title("Booster Position vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

plt.subplot(212)
plt.plot(t_array,data[1][1][1],'b',label="velocity")
plt.axvline(x=data[0],ls=':',color='g',label=r'$t_b$')
plt.axvline(x=data[1][0],ls='--',color='k',label=r'$t_f$')
plt.title("Booster Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()
