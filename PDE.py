import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class solution:
    def __init__(self, a, T_init, T_profile, Ox, Ot, dx, dt, n):
        
        self.a = a
        self.T0 = T_init
        self.Xmax = Ox
        self.Tmax = Ot
        self.dx = dx
        self.dt = dt
        self.n = n
        self.eta = a**2*dt/dx**2
        print(self.eta)
        self.Nx = int(Ox/dx+1)
        self.Nt = int(Ot/dt+1)
        self.T = np.zeros(shape=(self.Nt, self.Nx))
        self.start = 1
        self.T[0] = T_profile
        self.T[:,0] = T_profile[0] 
        self.T[:,-1] = T_profile[-1]

    def Fourier(self):
        self.T = np.zeros(shape=(self.Nt, self.Nx))
        xgrid, tgrid = np.meshgrid(np.linspace(0, self.Xmax, self.Nx), np.linspace(0, self.Tmax, self.Nt))
        L = float(self.Xmax)
        for i in range(int(self.n)):
            self.T += 4.0*self.T0*np.exp(-(self.a*np.pi*float(2*i+1)/L)**2*tgrid)\
            *np.sin(np.pi*float(2*i+1)*xgrid/L)/np.pi/(2*i+1)
        return self.T
    
    def explicit(self, n):
        for i in range(self.start, n+1):
            self.T[i,1:self.Nx-1] = self.T[i-1,1:self.Nx-1]+self.eta*(self.T[i-1,:self.Nx-2]\
                  + self.T[i-1,2:] - 2*self.T[i-1,1:self.Nx-1])
        self.start = n + 1
        
    def Crank_Nicolson(self, n): 
        alpha = np.zeros(self.Nx - 2)
        beta = np.zeros(self.Nx - 2)
        A = -np.ones(self.Nx - 2)
        C = np.ones(self.Nx - 2)*(2./self.eta + 2.)
        B = -np.ones(self.Nx - 2)

        for j in range(n):
            F = self.T[j, 0:self.Nx-2] + (2./self.eta - 2.)*self.T[j, 1:self.Nx-1] + self.T[j, 2:self.Nx]
            F[0] += self.T[j+1, 0]
            F[-1] += self.T[j+1, -1]
            alpha[0] = -B[0]/C[0]
            beta[0] = F[0]/C[0]
            for i in range(self.Nx - 3):              
                alpha[i + 1] = -B[i]/(A[i]*alpha[i] + C[i])
                beta[i + 1] = (F[i] - A[i]*beta[i])/(A[i]*alpha[i] + C[i])
            self.T[j+1][self.Nx-2] = (F[-1] - A[-1]*beta[-1])/(C[-1] + A[-1]*alpha[-1])
            for i in range(self.Nx - 3, 0, -1):
                self.T[j+1][i] = alpha[i]*self.T[j+1][i+1] + beta[i]    
                
            
    def plotter_3d(self, T, method):
        self.xgrid, self.tgrid = np.meshgrid(np.linspace(0, self.Xmax, self.Nx), np.linspace(0, self.Tmax, self.Nt))
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.contour(self.xgrid, self.tgrid, T, cmap=cm.magma, antialiased=False)
        ax.plot_wireframe(self.xgrid, self.tgrid, T, cmap=cm.magma, linewidth=0.5)       
        ax.set_zlim(0, self.T0)
        ax.set_xlim(0, self.Xmax)
        ax.set_ylim(0, self.Tmax)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_locator(LinearLocator(6))
        ax.yaxis.set_major_locator(LinearLocator(6))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_title(method)
        ax.view_init(30, 30)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()   
        
    def Newton(self, n):
        self.Te = 50.0
        self.h = 4.0
        for i in range(self.start, n + 1):
            self.T[i,1:self.Nx-1] = self.T[i-1,1:self.Nx-1]\
            + self.eta*(self.T[i-1,:self.Nx-2] + self.T[i-1,2:] - 2*self.T[i-1,1:self.Nx-1])\
            - self.h*dt*(self.T[i-1,1:self.Nx-1] - self.Te)
        self.start = n + 1        
    
    def plotter_2d(self, method):
        xgrid, tgrid = np.meshgrid(np.linspace(0, self.Xmax, self.Nx), np.linspace(0, self.Tmax, self.Nt))
        fig, ax = plt.subplots(figsize=(10,6))
        plot = ax.contourf(xgrid, tgrid, self.T, cmap=cm.magma, levels = np.linspace(0, self.T0, 11))
        ax.set_title(method)
        plt.xlabel('x')
        plt.ylabel('t')
        fig.colorbar(plot, shrink=1, aspect=10)
        plt.show() 
    
    def run(self, method, count):
        if method == 'Fourier':
            self.plotter_3d(self.Fourier(), method)
#            for n in count:   
#                self.Fourier()
#                self.plotter_2d(method)
        elif method == 'Explicit':
            k = 0
            for n in count:
                k = n - k    
                self.explicit(k)
                self.plotter_2d(method)
        elif method == 'Crank-Nicolson':
            k = 0
            for n in count:
                k = n - k    
                self.Crank_Nicolson(k)
                self.plotter_2d(method)
        elif method == 'Newton':
            k = 0
            for n in count:
                k = n - k    
                self.Newton(k)
                self.plotter_2d(method)            
                

T_init = 100.0       
(a, Ox, Ot, dx, dt, n) = (1, 2.0, 1.0, 0.02, 0.002, 500.0)
T_prof = np.ones(int(Ox/dx + 1))*T_init
T_prof[0] = T_prof[-1]  = 0  
steps = [int(Ot/dt)]    
#obj = solution(a, T_init, T_prof, Ox, Ot, dx, dt, n)
#obj.run('Fourier', steps)

(a, Ox, Ot, dx, dt, n) = (0.3, 2.0, 1.0, 0.02, 0.002, 500.0)
obj1 = solution(a, T_init, T_prof, Ox, Ot, dx, dt, n)
obj1.run('Explicit', steps)

print('initial distribution is sin')
T_prof1 = T_init*np.sin(np.pi*np.linspace(0, Ox, int(Ox/dx + 1))/Ox)
T_prof1[0] = T_prof[-1]  = 0
(a, Ox, Ot, dx, dt, n) = (0.3, 2.0, 1.0, 0.02, 0.002, 500.0)
obj2 = solution(a, T_init, T_prof1, Ox, Ot, dx, dt, n)
obj2.run('Explicit', steps)

(a, Ox, Ot, dx, dt, n) = (0.3, 2.0, 1.0, 0.02, 0.002, 500.0)
obj3 = solution(a, T_init, T_prof, Ox, Ot, dx, dt, n)
obj3.run('Crank-Nicolson', steps)

T_prof2 = np.ones(int(Ox/dx + 1))*T_init
T_prof2[:int((Ox/dx + 1)/2)] = T_init/2
T_prof2[0] = T_prof2[-1] = 0
obj4 = solution(a, T_init, T_prof2, Ox, Ot, dx, dt, n)
obj4.run('Crank-Nicolson', steps)

(a, Ox, Ot, dx, dt, n) = (0.3, 2.0, 1.0, 0.02, 0.002, 500.0)
T_prof3 = np.ones(int(Ox/dx + 1))*T_init
T_prof3[0] = T_prof[-1]  = 0
obj5 = solution(a, T_init, T_prof3, Ox, Ot, dx, dt, n)
obj5.run('Newton', steps)

(a, Ox, Ot, dx, dt, n) = (0.3, 2.0, 1.0, 0.02, 0.002, 500.0)
T_prof4 = np.ones(int(Ox/dx + 1))*T_init
T_prof4[:int((Ox/dx + 1)/2)] = T_init/2
T_prof4[0] = T_prof2[-1] = 0
obj6 = solution(a, T_init, T_prof4, Ox, Ot, dx, dt, n)
obj6.run('Newton', steps)