import numpy as np
import rcwa
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

mask = (x<0)

thickness = np.linspace(2,2,1)
efficiency = []

for t in thickness:
    layers = [
        rcwa.Layer(n=1),
        rcwa.Layer(n=1, t=1),
        rcwa.Layer(n=np.where(mask, 1, 1.4482+7.5367j), t=t),
        rcwa.Layer(n=1.4482+7.5367j, t=1),
        rcwa.Layer(n=1.4482+7.5367j),
    ]

    ## Step 2 define modes -------------
    AOI = np.radians(0)#thetai
    POI = np.radians(0)#phi

    modes = rcwa.Modes(
        wavelength=0.6328,
        kx0=0,
        ky0=0,
        period_x=4,
        period_y=2,
        harmonics_x=10,
        harmonics_y=0
    )

    modes.set_direction(
        kx0=np.cos(POI) * np.sin(AOI)*layers[0].n,
        ky0=np.sin(POI) * np.sin(AOI)*layers[0].n
    )

    ## Step 3 run your simulation -------------
    simulation = rcwa.Simulation(
        modes=modes,
        layers=layers,
        keep_modes=True
    )

    R, T = simulation.run(Ex=0, Ey=1).get_efficiency()
    mid=len(R)//2
    efficiency.append([R[mid-6], R[mid-5], R[mid-4], R[mid-3],R[mid-2],R[mid-1],R[mid]])


# Step 4 visualize the results -------------
length=len(efficiency[0])
x=np.linspace(-6,0,7,dtype=int)
plt.plot(x, efficiency[0])
plt.xlabel('Diffraction order')
plt.ylabel('Efficiency')
plt.title('Efficiency vs Thickness')
plt.legend()
plt.show()