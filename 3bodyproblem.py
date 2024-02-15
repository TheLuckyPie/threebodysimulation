#Import Packages
import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
plt.style.use("Solarize_Light2")
year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
hour, minute, second = datetime.now().hour, datetime.now().minute, datetime.now().second

#Constants
G = 6.67430e-11
M_sun = 1.989e+30
Earth_Sun_Distance = 1.49597e11
Earth_Sun_vel = 29784.8
Year_s = 365*24*60*60

#Normalized Non-dimensional Constants
m_r = M_sun
r_r = Earth_Sun_Distance * 10
v_r = Earth_Sun_vel
t_r = Year_s * 100

#Defining Random Position and Velocity Vectors
rand_mass = np.random.uniform(0.8, 1.2, [1,3])
rand_pos = np.random.uniform(-0.5,0.5,[1,9])
rand_vel = np.random.uniform(-0.1,0.1,[1,9])

#Defining Constant for Dimensionless E.O.M.
C_1 = (G * m_r * t_r)/(r_r**2 * v_r)
C_2 = (v_r * t_r) / (r_r)

#Defining Intial Conditions (Mass, Position, Velocity)
m1, m2, m3 = rand_mass[0,0], rand_mass[0,1], rand_mass[0,2]  #Masses in terms of m_r
r1, r2, r3 = rand_pos[0,0:3], rand_pos[0,3:6], rand_pos[0,6:9]
v1, v2, v3 = rand_vel[0,0:3], rand_vel[0,3:6], rand_vel[0,6:9]

intial_cond_array = np.concatenate((r1,r2,r3,v1,v2,v3))
time_span = np.linspace(0,5,5000) 

#Defining Function which creates E.O.M. given intial condtions:#
def ThreeBodyEquations(pos_vel_array,t):
    r_1, r_2, r_3 = pos_vel_array[:3], pos_vel_array[3:6], pos_vel_array[6:9]
    v_1, v_2, v_3 = pos_vel_array[9:12], pos_vel_array[12:15], pos_vel_array[15:18] 
    r_12=sci.linalg.norm(r_2-r_1) #Calculate magnitude or norm of vector    dv1=K1*m2*(r2-r1)/r**3
    r_13=sci.linalg.norm(r_3-r_1)
    r_23=sci.linalg.norm(r_3-r_2)
    dv1=C_1*m2*(r_2-r_1)/r_12**3 + C_1*m3*(r_3 - r_1) / (r_13**3)
    dv2=C_1*m1*(r_1-r_2)/r_12**3 + C_1*m3*(r_3 - r_2) / (r_23**3)
    dv3=C_1*m1*(r_1-r_3)/(r_13**3) + C_1*m2*(r_2 - r_3) / (r_23**3)
    dr1=C_2*v_1
    dr2=C_2*v_2
    dr3=C_2*v_3    
    return np.concatenate((dr1, dr2, dr3, dv1, dv2, dv3))

#Package initial parameters and get solution
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,intial_cond_array,time_span)
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]


# Create Animated Plot
fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111, projection="3d")
orbit_line1, = ax.plot([], [], [], color="blue")
orbit_line2, = ax.plot([], [], [], color="red")
orbit_line3, = ax.plot([], [], [], color="green")
star1_point, = ax.plot([], [], [], 'o', color="darkblue", markersize=10)
star2_point, = ax.plot([], [], [], 'o', color="darkred", markersize=10)
star3_point, = ax.plot([], [], [], 'o', color="darkgreen", markersize=10)
plt.rcParams.update({'font.size': 24})

# Function to initialize the plot
def init():
    ax.set_xlabel("x-coordinate [10 AU]", fontsize=14)
    ax.set_ylabel("y-coordinate [10 AU]", fontsize=14)
    ax.set_zlabel("z-coordinate [10 AU]", fontsize=14)
    ax.set_title("Orbits of a Three-body system\n", fontsize=14)
    ax.legend([f"Mass 1: {m1:.3f} Solar Masses", f"Mass 2: {m2:.3f} Solar Masses", f"Mass 3: {m3:.3f} Solar Masses"], loc="upper left", fontsize=14)
    ax.view_init(elev=10, azim=0)  # Set the initial view angle
    return orbit_line1, orbit_line2, orbit_line3, star1_point, star2_point, star3_point

# Function to update the animation
def update(frame):
    orbit_line1.set_data(r1_sol[:frame, 0], r1_sol[:frame, 1])
    orbit_line1.set_3d_properties(r1_sol[:frame, 2])
    orbit_line2.set_data(r2_sol[:frame, 0], r2_sol[:frame, 1])
    orbit_line2.set_3d_properties(r2_sol[:frame, 2])
    orbit_line3.set_data(r3_sol[:frame, 0], r3_sol[:frame, 1])
    orbit_line3.set_3d_properties(r3_sol[:frame, 2])
    star1_point.set_data(r1_sol[frame, 0], r1_sol[frame, 1])
    star1_point.set_3d_properties([r1_sol[frame, 2]])
    star2_point.set_data(r2_sol[frame, 0], r2_sol[frame, 1])
    star2_point.set_3d_properties([r2_sol[frame, 2]])
    star3_point.set_data(r3_sol[frame, 0], r3_sol[frame, 1])
    star3_point.set_3d_properties([r3_sol[frame, 2]])

    # Slice the data arrays up to the current frame
    r1_frame = r1_sol[:frame+1]
    r2_frame = r2_sol[:frame+1]
    r3_frame = r3_sol[:frame+1]

    # Calculate the minimum and maximum values for each axis dynamically
    min_x = min(np.min(r1_frame[:, 0]), np.min(r2_frame[:, 0]), np.min(r3_frame[:, 0]))
    max_x = max(np.max(r1_frame[:, 0]), np.max(r2_frame[:, 0]), np.max(r3_frame[:, 0]))
    min_y = min(np.min(r1_frame[:, 1]), np.min(r2_frame[:, 1]), np.min(r3_frame[:, 1]))
    max_y = max(np.max(r1_frame[:, 1]), np.max(r2_frame[:, 1]), np.max(r3_frame[:, 1]))
    min_z = min(np.min(r1_frame[:, 2]), np.min(r2_frame[:, 2]), np.min(r3_frame[:, 2]))
    max_z = max(np.max(r1_frame[:, 2]), np.max(r2_frame[:, 2]), np.max(r3_frame[:, 2]))
    
    # Set the axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    # Set the new view angle for the plot
    ax.view_init(elev=10, azim=frame/5)

    return orbit_line1, orbit_line2, orbit_line3, star1_point, star2_point, star3_point

# Number of frames to animate (number of time steps)
num_frames = min(len(r1_sol), len(r2_sol), len(r3_sol))

#Animate and save the file
ani = FuncAnimation(fig, update, frames=3600, init_func=init, interval=10, blit=True)
ani.save(f"threebodypythonanimation_{year}_{month}_{day}__{hour}_{minute}_{second}.mp4", fps=60)
plt.ioff()
plt.show()
