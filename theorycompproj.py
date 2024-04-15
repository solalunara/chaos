import matplotlib.pyplot as plt;
import numpy as np;
from matplotlib.widgets import Button, Slider;
from typing import Callable;


def plot_multiple_datasets( xdata: np.ndarray, ydata: np.ndarray, ax: plt.Axes, plotfn: Callable, **kwargs ):
    """graphs multiple datasets on the same plot

    Args:
        xdata (array, dim 1 or 2): EITHER a 1D array of x values for all graphs, OR a 2D array of x values for each graph (Point Index, Graph Index)
        ydata (array, dim 1 or 2): EITHER a 1D array of y values for all graphs, OR a 2D array of y values for each graph (Point Index, Graph Index)
        ax (plt.Axes): axis to plot the graphs on
        plotfn (Callable): function to plot the data with (e.g. ax.plot, ax.scatter, etc.)
        
    Optional Args:
        title (string): title of the plot
        xlabel (string): label to show on the x axis
        ylabel (string): label to show on the y axis
        xmin (float): minimum value of x axis
        xmax (float): maximum value of x axis
        ymin (float): minimum value of y axis
        ymax (float): maximum value of y axis

    Raises:
        ValueError: if x or y data has the wrong dimension
        ValueError: if x and y data hold data for different numbers of graphs

    Returns:
        graphs: an array of the graphs that are plotted
    """
    title = kwargs.get( 'title', None );
    xlabel = kwargs.get( 'xlabel', None );
    ylabel = kwargs.get( 'ylabel', None );
    xmin = kwargs.get( 'xmin', None );
    xmax = kwargs.get( 'xmax', None );
    ymin = kwargs.get( 'ymin', None );
    ymax = kwargs.get( 'ymax', None );
    
    graphs = [];
    
    if ydata.ndim == 1:
        ydata = ydata.reshape( (len(ydata), 1) );
    elif ydata.ndim != 2: raise ValueError( "ydata must have dimension 1 or 2" );
    
    if xdata.ndim == 1:
        xdata = xdata.reshape( (len(xdata), 1) );
    elif xdata.ndim != 2: raise ValueError( "xdata must have dimension 1 or 2" );
    
    if len( ydata[ 0 ] ) != len( xdata[ 0 ] ): 
        if len( ydata[ 0 ] ) == 1:
            ydata = np.repeat( ydata, len( xdata ), axis=1 );
        elif len( xdata[ 0 ] ) == 1:
            xdata = np.repeat( xdata, len( ydata ), axis=1 );
        else: raise ValueError( "xdata and ydata must have the same length" );
        
    unpackable = True;
    for i in range( len( ydata[ 0 ] ) ):
        graphs.append( None );
        collection = plotfn( xdata[ :, i ], ydata[ :, i ] );
        if unpackable:
            try:
                graphs[ i ], = collection;
            except TypeError:
                unpackable = False;
                graphs[ i ] = collection;
        else:
            graphs[ i ] = collection;
        
    
    if title: ax.set_title( title );
    if xlabel: ax.set_xlabel( xlabel );
    if ylabel: ax.set_ylabel( ylabel );
    
    if xmin is not None: ax.set_xlim( xmin=xmin );
    if xmax is not None: ax.set_xlim( xmax=xmax );
    if ymin is not None: ax.set_ylim( ymin=ymin );
    if ymax is not None: ax.set_ylim( ymax=ymax );
    
    return graphs;
    
    
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
    angles[ i ] = ( angles[ i - 1 ] + angular_velocities[ i ] * dt + np.pi ) % ( 2 * np.pi ) - np.pi;

lyapunov_diff = np.average( np.abs( np.diff( angles, axis=1 ) ), axis=1 );

poincare_x = angles[ :-1:, :, :, : ];
poincare_y = angles[  1::, :, :, : ];

fig = plt.figure();
gs = fig.add_gridspec(2, 3,  width_ratios=(1, 1, 10), height_ratios=(1, 1),
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0.45, hspace=0.45)
ax_lines = fig.add_subplot( gs[0, 2] );
ax_scatter = fig.add_subplot( gs[1, 2] );
ax_force = fig.add_subplot( gs[:, 0] );
ax_freq = fig.add_subplot( gs[:, 1] );

lines = plot_multiple_datasets( time, lyapunov_diff[ :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], ax_lines, ax_lines.plot, title=f"difference plot for R={R}, b={b}, M={M}", xlabel='time', ylabel='difference', xmin=0, xmax=N*dt, ymin=0, ymax=2*np.pi );

poincare_graphs = plot_multiple_datasets( poincare_x[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], poincare_y[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], ax_scatter, ax_scatter.scatter, title=f"poincare plot for R={R}, b={b}, M={M}", xlabel='x', ylabel='y', xmin=-np.pi, xmax=np.pi, ymin=-np.pi, ymax=np.pi );

f_slider = Slider(
    ax=ax_force,
    label="force amplitude",
    valmin=fmin,
    valmax=fmax,
    valinit=F_init,
    orientation="vertical"
);

w_slider = Slider(
    ax=ax_freq,
    label="frequency",
    valmin=wmin,
    valmax=wmax,
    valinit=W_init,
    orientation="vertical"
);

def update(val):
    f = f_slider.val;
    w = w_slider.val;
    lines[ 0 ].set_ydata( lyapunov_diff[ :, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ] );
    for i in range( X ):
        poincare_graphs[ i ].set_offsets( np.column_stack( (poincare_x[ :, i, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ], poincare_y[ :, i, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ]) ) );
    fig.canvas.draw_idle();

f_slider.on_changed(update);
w_slider.on_changed(update);

    
#plot_multiple_lines( np.linspace( 0, N * dt, N ), angles, f"pendulum plot for R={R}, b={b}, M={M}, F={F}, w={w}" );
plt.show();
