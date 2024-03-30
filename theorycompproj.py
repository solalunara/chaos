import matplotlib.pyplot as plt;
import numpy as np;
import matplotlib.animation as animation;


lines = [];
text = None;
fig = None;
ax = None;
def plot_multiple_lines(xdata, ydata, title, xlabel, ylabel):
    global fig, ax, lines, text;
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if xdata.ndim == 1:
        for i in range(len(ydata[0])):
            lines.append( None );
            lines[ i ], = ax.plot(xdata, ydata[:,i])
    elif xdata.ndim == 2:
        for i in range(len(ydata[0])):
            lines.append( None );
            lines[ i ], = ax.plot(xdata[:,i], ydata[:,i])
    else: raise ValueError("xdata must have dimension 1 or 2");
    
    text = plt.text(0, 17, 'Hello');
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title( title )
    ax.set_ylim([-20, 20])
    ax.set_xlim([-20, 20])
    

N = 5000;
dt = 0.1;
X = 10;
W = 3000;
initial_angles = np.linspace( -0.001, 0.001, X );
initial_angular_velocities = [ 0.5 for _ in range( X ) ];

g = 9.81;
R = 10;
b = 0.1;
M = 1;
F_arr = np.linspace( 0, 10, 5 );
w_arr = np.reshape( np.linspace( 0, 2, W ), (1,W) );

def Acceleration( time, angle, angular_velocity, w ):
    return -g/R * np.sin( angle ) - b/M * angular_velocity + F * np.cos( w * time );


for F in F_arr:
    time = np.linspace( 0, N * dt, N );
    angles = np.zeros( (N,X,W) );
    angular_velocities = np.zeros( (N,X,W) );
    for i in range( W ):
        angles[ 0, :, i ] = initial_angles;
        angular_velocities[ 0, :, i ] = initial_angular_velocities;
    
    for i in range( 1, N ):
        k_1 = Acceleration( time[ i - 1 ], angles[ i - 1 ], angular_velocities[ i - 1 ], w_arr );
        k_2 = Acceleration( time[ i - 1 ] + dt/2, angles[ i - 1 ] + k_1 * dt/2, angular_velocities[ i - 1 ] + k_1 * dt/2, w_arr );
        k_3 = Acceleration( time[ i - 1 ] + dt/2, angles[ i - 1 ] + k_2 * dt/2, angular_velocities[ i - 1 ] + k_2 * dt/2, w_arr );
        k_4 = Acceleration( time[ i - 1 ] + dt, angles[ i - 1 ] + k_3 * dt, angular_velocities[ i - 1 ] + k_3 * dt, w_arr );
        angular_velocities[ i ] = angular_velocities[ i - 1 ] + ( k_1 + 2*k_2 + 2*k_3 + k_4 ) * dt/6;
        angles[ i ] = angles[ i - 1 ] + angular_velocities[ i ] * dt;
        
    plot_multiple_lines( angles[ :, :, 0 ], angular_velocities[ :, :, 0 ], f"pendulum plot for R={R}, b={b}, M={M}, F={F}", 'angle', 'angular velocity' );
    anims = [];
    def Animate( frame ):
        for j in range( X ):
            lines[ j ].set_xdata( angles[ :, j, frame ] );
            lines[ j ].set_ydata( angular_velocities[ :, j, frame ] );
        text.set_text( f"w: {w_arr[ 0, frame ]}" );
        arr = lines;
        arr.append( text );
        return arr;
    ani = animation.FuncAnimation( fig, Animate, frames=W, interval=100 );
    #plot_multiple_lines( np.linspace( 0, N * dt, N ), angles, f"pendulum plot for R={R}, b={b}, M={M}, F={F}, w={w}" );
    plt.tight_layout()
    plt.show()
    