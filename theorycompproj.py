import matplotlib.pyplot as plt;
import numpy as np;
import matplotlib.animation as animation;
from matplotlib.widgets import Button, Slider;


lines = [];
text = None;
fig = None;
ax = None;
def plot_multiple_lines(xdata, ydata, title, xlabel, ylabel):
    global fig, ax, lines, text;
    fig, ax = plt.subplots( figsize=(10, 6) );
    
    if ydata.ndim == 1:
        if xdata.ndim == 1:
            lines.append( None );
            lines[ 0 ], = ax.plot(xdata, ydata)
        elif xdata.ndim == 2:
            for i in range(len(xdata[0])):
                lines.append( None );
                lines[ i ], = ax.plot(xdata[:,i], ydata)
        else: raise ValueError("xdata must have dimension 1 or 2");
    elif ydata.ndim == 2:
        if xdata.ndim == 1:
            for i in range(len(ydata[0])):
                lines.append( None );
                lines[ i ], = ax.plot(xdata, ydata[:,i])
        elif xdata.ndim == 2:
            for i in range(len(ydata[0])):
                lines.append( None );
                lines[ i ], = ax.plot(xdata[:,i], ydata[:,i])
        else: raise ValueError("xdata must have dimension 1 or 2");
    else: raise ValueError("ydata must have dimension 1 or 2");
    
    ax.set_xlabel( xlabel );
    ax.set_ylabel( ylabel );
    ax.set_title( title );
    ax.set_ylim( [0, 1000] );
    ax.set_xlim( [0, 500] );
    fig.subplots_adjust( left=0.25 );
    
def find_nearest( array, value ):
    array = np.asarray( array );
    idx = (np.abs( array - value )).argmin();
    return idx;

N = 5000; # number of time steps
dt = 0.1; # time step
X = 10;   # number of pendulums to simulate
F = 50;  # number of forces to simulate, also acts as the number of frames in the animation
W = 50;  # number of angular velocities to simulate
fmin = 0;
fmax = 3;
wmin = 0;
wmax = 3;
initial_angles = np.linspace( -0.001, 0.001, X ) + 0.5;
initial_angular_velocities = np.array( [ 0 for _ in range( X ) ] );


g = 9.81;
R = 10;
b = 0.1;
M = 1;
F_arr = np.reshape( np.linspace( fmin, fmax, F ), (1,F,1) );
W_arr = np.reshape( np.linspace( wmin, wmax, W ), (1,1,W) );
F_init = 0.5;
W_init = 0.5;


def Acceleration( time, angle, angular_velocity, F, w ):
    return -g/R * np.sin( angle ) - b/M * angular_velocity + F * np.cos( w * time );


time = np.linspace( 0, N * dt, N );
angles = np.zeros( (N,X,F,W) );
angular_velocities = np.zeros( (N,X,F,W) );
angles[ 0, :, :, : ] = initial_angles[ :, None, None ];
angular_velocities[ 0, :, :, : ] = initial_angular_velocities[ :, None, None ];

for i in range( 1, N ):
    k_1 = Acceleration( time[ i - 1 ], angles[ i - 1 ], angular_velocities[ i - 1 ], F_arr, W_arr );
    k_2 = Acceleration( time[ i - 1 ] + dt/2, angles[ i - 1 ] + k_1 * dt/2, angular_velocities[ i - 1 ] + k_1 * dt/2, F_arr, W_arr );
    k_3 = Acceleration( time[ i - 1 ] + dt/2, angles[ i - 1 ] + k_2 * dt/2, angular_velocities[ i - 1 ] + k_2 * dt/2, F_arr, W_arr );
    k_4 = Acceleration( time[ i - 1 ] + dt, angles[ i - 1 ] + k_3 * dt, angular_velocities[ i - 1 ] + k_3 * dt, F_arr, W_arr );
    angular_velocities[ i ] = angular_velocities[ i - 1 ] + ( k_1 + 2*k_2 + 2*k_3 + k_4 ) * dt/6;
    angles[ i ] = angles[ i - 1 ] + angular_velocities[ i ] * dt;

diff = np.abs( np.diff( angles, axis=1 ) );
    
plot_multiple_lines( time, diff[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], f"pendulum plot for R={R}, b={b}, M={M}", 'time', 'difference' );

axforce = fig.add_axes([0.15, 0.2, 0.0225, 0.63])
f_slider = Slider(
    ax=axforce,
    label="force amplitude",
    valmin=fmin,
    valmax=fmax,
    valinit=F_init,
    orientation="vertical"
)

axfreq = fig.add_axes([0.05, 0.2, 0.0225, 0.63])
w_slider = Slider(
    ax=axfreq,
    label="frequency",
    valmin=wmin,
    valmax=wmax,
    valinit=W_init,
    orientation="vertical"
)

def update(val):
    for i in range( X - 1 ):
        lines[ i ].set_ydata( diff[ :, i, find_nearest( F_arr, f_slider.val ), find_nearest( W_arr, w_slider.val ) ] );
    fig.canvas.draw_idle();
f_slider.on_changed(update);
w_slider.on_changed(update);

    
#plot_multiple_lines( np.linspace( 0, N * dt, N ), angles, f"pendulum plot for R={R}, b={b}, M={M}, F={F}, w={w}" );
plt.show();
