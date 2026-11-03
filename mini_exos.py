import numpy as np
import scipy
import matplotlib.pyplot as plt

### Paramètres

f=196
T=1/f
w=2*np.pi*f
t0,t1=0,5*T
y0=[1,0]

###

def rhs(t,y):
    return np.array([y[1],-(w**2)*y[0]])

sol=scipy.integrate.solve_ivp(rhs, (t0, t1), y0, method="RK45",t_eval=np.linspace(t0,t1,1000))

plt.plot(sol.t,sol.y[0])
plt.xlabel('temps (s)')
plt.grid()
plt.title('U en fonction du temmps')
plt.show()
