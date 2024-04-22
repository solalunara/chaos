import matplotlib.pyplot as plt;
import numpy as np;
from matplotlib.widgets import Button, Slider;
from typing import Callable;
import matplotlib.animation as animation;


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
        s (float): size of the points on a scatter graph

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
    s = kwargs.get( 's', None );
    
    graphs = [];
    
    rectangular = True;
    try:
        #assume the arrays are numpy (rectangular)
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
            
    except AttributeError:
        #handle the case where xdata and ydata are a list of arrays
        rectangular = False;
        if len( xdata ) != len( ydata ): raise ValueError( "xdata and ydata must have the same length" );
         
    unpackable = True;
    data_length = len( ydata[ 0 ] ) if rectangular else len( ydata );   
    for i in range( data_length ):
        graphs.append( None );
        collection = None;
        if s is not None:
            collection = plotfn( xdata[ :, i ], ydata[ :, i ], s=s ) if rectangular else plotfn( xdata[ i ], ydata[ i ], s=s );
        else:
            collection = plotfn( xdata[ :, i ], ydata[ :, i ] ) if rectangular else plotfn( xdata[ i ], ydata[ i ] );
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
    
    #float value of zero is False, so we need to check if the value is None
    if xmin is not None: ax.set_xlim( xmin=xmin );
    if xmax is not None: ax.set_xlim( xmax=xmax );
    if ymin is not None: ax.set_ylim( ymin=ymin );
    if ymax is not None: ax.set_ylim( ymax=ymax );
    
    return graphs;

def poincare_recurrence( phase: np.ndarray, error: np.ndarray ):
    """generates a list of phase space lines that recur to within a certain threshold by truncating the dynamical evolution of a system

    Args:
        phase (np.ndarray): 3D array:
                                first dimension must be length 2, representing the phase space of the system
                                second dimension must be the time steps
                                third dimension must be the number of pendulums
        error (np.ndarray): 3D array:
                                first dimension must be length 2, representing the error in the phase space of the system
                                second dimension must be the time steps
                                third dimension must be the number of pendulums
        threshold (float): the float value precision to check for recurrence
    
    Returns:
        lines_x: a non-rectangular list of the x part of phase space lines (numpy array) that recur
        lines_y: a non-rectangular list of the y part of phase space lines (numpy array) that recur
    
    Raises:
        ValueError: if no recurrence is found
    """
    lines_x = [];
    lines_y = [];
    for phase_plot in range( phase.shape[ 2 ] ):
        initial_condition = phase[ :, 0, phase_plot ];
        for phase_point in range( 1, phase.shape[ 1 ] ):
            if np.abs( phase[ 0, phase_point, phase_plot ] - initial_condition[ 0 ] ) < error[ 0, phase_point, phase_plot ] and np.abs( phase[ 1, phase_point, phase_plot ] - initial_condition[ 1 ] ) < error[ 1, phase_point, phase_plot ]:
                lines_x.append( phase[ 0, :phase_point+1:, phase_plot ] );
                lines_y.append( phase[ 1, :phase_point+1:, phase_plot ] );
                break;
    if len( lines_x ) == 0: raise ValueError( "no recurrence found" );
    return lines_x, lines_y;
    
def find_nearest( array, value ):
    array = np.asarray( array );
    idx = (np.abs( array - value )).argmin();
    return idx;

N = 10000; # number of time steps
dt = 0.03; # time step
A = 3;    # number of initial angles
B = 3;    # number of initial angular velocities
X = A*B;  # number of pendulums to simulate
F = 10;   # number of forces to simulate, also acts as the number of frames in the animation
W = 10;   # number of angular velocities to simulate
fmin = 0;
fmax = 3;
wmin = 0;
wmax = 3;
initial_angles = np.linspace( -0.001, 0.001, A ) + 0.5;
initial_angular_velocities = np.linspace( -0.001, 0.001, B );

initial_angles_mesh = np.repeat( initial_angles, B );
initial_angular_velocities_mesh = np.resize( initial_angular_velocities, X )


g = 9.81;
R = 10;
b = 0.1;
M = 1;
F_arr = np.reshape( np.linspace( fmin, fmax, F ), (1,F,1) );
W_arr = np.reshape( np.linspace( wmin, wmax, W ), (1,1,W) );
F_init = 0.5;
W_init = 0.5;
T_init = 0;


def Acceleration( time, angle, angle_err, angular_velocity, angular_velocity_err, F, w ):
    acceleration = -g/R * np.sin( angle ) - b/M * angular_velocity + F * np.cos( w * time );
    error = np.sqrt( (g/R * np.cos( angle ) * angle_err)**2 + (b/M * angular_velocity_err)**2 );
    return acceleration, error;

def RK4_Coefficient( time, angle, angle_err, angular_velocity, angular_velocity_err, F, w, dt ):
    k_1, k1_err = Acceleration( time, angle, angle_err, angular_velocity, angular_velocity_err, F, w );
    k_2, k2_err = Acceleration( time + dt/2, angle + k_1 * dt/2, np.sqrt( angle_err**2 + (dt/2 * k1_err)**2 ), angular_velocity + k_1 * dt/2, np.sqrt( angular_velocity_err**2 + (dt/2 * k1_err)**2 ), F, w );
    k_3, k3_err = Acceleration( time + dt/2, angle + k_2 * dt/2, np.sqrt( angle_err**2 + (dt/2 * k2_err)**2 ), angular_velocity + k_2 * dt/2, np.sqrt( angular_velocity_err**2 + (dt/2 * k2_err)**2 ), F, w );
    k_4, k4_err = Acceleration( time + dt, angle + k_3 * dt, np.sqrt( angle_err**2 + (dt * k3_err)**2 ), angular_velocity + k_3 * dt, np.sqrt( angular_velocity_err**2 + (dt * k3_err)**2 ), F, w );
    rk4 = ( k_1 + 2*k_2 + 2*k_3 + k_4 ) * dt/6;
    rk4_err = np.sqrt( k1_err**2 + 4*k2_err**2 + 4*k3_err**2 + k4_err**2 ) * dt/6;
    return rk4, rk4_err;
    


time = np.linspace( 0, N * dt, N );
angles = np.zeros( (N,X,F,W) );
angles_error = np.zeros( (N,X,F,W) );
angular_velocities = np.zeros( (N,X,F,W) );
angular_velocities_error = np.zeros( (N,X,F,W) );
angles[ 0, :, :, : ] = initial_angles_mesh[ :, None, None ];
angular_velocities[ 0, :, :, : ] = initial_angular_velocities_mesh[ :, None, None ];

for i in range( 1, N ):
    rk4, rk4_err = RK4_Coefficient( time[ i - 1 ], angles[ i - 1 ], angles_error[ i - 1 ], angular_velocities[ i - 1 ], angular_velocities_error[ i - 1 ], F_arr, W_arr, dt );
    angular_velocities[ i ] = angular_velocities[ i - 1 ] + rk4;
    angles[ i ] = angles[ i - 1 ] + angular_velocities[ i ] * dt;
    
    #estimate inherent error on RK4 and euler
    doublestep_estimation_0_rk4 = angular_velocities[ i ] + RK4_Coefficient( time[ i ], angles[ i ], angles_error[ i ], angular_velocities[ i ], angular_velocities_error[ i ], F_arr, W_arr, dt )[ 0 ];
    doublestep_estimation_1_rk4 = angular_velocities[ i - 1 ] + rk4;
    doublestep_estimation_0_euler = angles[ i ] + angular_velocities[ i ] * dt;
    doublestep_estimation_1_euler = angles[ i - 1 ] + angular_velocities[ i - 1 ] * dt * 2;
    angular_velocities_error[ i ] = np.sqrt( ( np.abs( doublestep_estimation_0_rk4 - doublestep_estimation_1_rk4 ) / 15 )**2 + rk4_err**2 + angular_velocities_error[ i - 1 ]**2 );
    angles_error[ i ] = np.sqrt( ( np.abs( doublestep_estimation_0_euler - doublestep_estimation_1_euler ) )**2 + angles_error[ i - 1 ]**2 + dt**2 * angular_velocities_error[ i - 1 ]**2 );
    
angles = np.mod( angles + np.pi, 2 * np.pi ) - np.pi;

lyapunov_diff = np.average( np.abs( np.diff( angles, axis=1 ) ), axis=1 );

poincare_x = angles[ :-1:, :, :, : ];
poincare_y = angles[  1::, :, :, : ];

phase = np.zeros( (2,N,X,F,W) );
phase[ 0 ] = angles;
phase[ 1 ] = angular_velocities;
error_phase = np.zeros( (2,N,X,F,W) );
error_phase[ 0 ] = angles_error;
error_phase[ 1 ] = angular_velocities_error;

poincare_recurrence_lines_x, poincare_recurrence_lines_y = poincare_recurrence( phase[ :, :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], error_phase[ :, :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ] );

fig = plt.figure( figsize=(12, 8) );
gs = fig.add_gridspec(4, 5,  width_ratios=(1, 1, 10, 10, 10), height_ratios=(1, 1, 1, 1),
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0.45, hspace=0.45)
ax_lines = fig.add_subplot( gs[ :2, 2 ] );
ax_scatter = fig.add_subplot( gs[ 2:, 2 ] );
ax_force = fig.add_subplot( gs[ :, 0 ] );
ax_freq = fig.add_subplot( gs[ :, 1 ] );
ax_phase = fig.add_subplot( gs[ :, 3 ] );
ax_poincaremaps = [];
ax_poincaremaps.append( fig.add_subplot( gs[ 0, 4 ] ) );
ax_poincaremaps.append( fig.add_subplot( gs[ 1, 4 ] ) );
ax_poincaremaps.append( fig.add_subplot( gs[ 2, 4 ] ) );
ax_poincaremaps.append( fig.add_subplot( gs[ 3, 4 ] ) );

lines = plot_multiple_datasets( time, angles_error[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], ax_lines, ax_lines.plot, title=f"error plot for R={R}, b={b}, M={M}", xlabel='time', ylabel='error (radians)', xmin=0, xmax=N*dt, ymin=0 );
poincare_graphs = plot_multiple_datasets( poincare_x[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], poincare_y[ :, :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], ax_scatter, ax_scatter.scatter, title=f"poincare plot for R={R}, b={b}, M={M}", xlabel='x', ylabel='y', xmin=-np.pi, xmax=np.pi, ymin=-np.pi, ymax=np.pi, s=1 );
phase_space_graphs = plot_multiple_datasets( angles[ find_nearest( time, T_init ), :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], angular_velocities[ find_nearest( time, T_init ), :, find_nearest( F_arr, F_init ), find_nearest( W_arr, W_init ) ], ax_phase, ax_phase.scatter, title=f"phase space plot for R={R}, b={b}, M={M}", xlabel='angle', ylabel='angular velocity', xmin=-np.pi, xmax=np.pi, ymin=-8, ymax=8, s=1 );
poincare_maps = [];
for i in range( len( ax_poincaremaps ) ):
    poincare_maps.append( plot_multiple_datasets( poincare_recurrence_lines_x[ i ], poincare_recurrence_lines_y[ i ], ax_poincaremaps[ i ], ax_poincaremaps[ i ].scatter, title=f"poincare map for theta={angles[ 0, i, 0, 0 ]:.2f}, w_0={angular_velocities[ 0, i, 0, 0 ]:.2f}", xlabel='angle', ylabel='angular velocity', xmin=-np.pi, xmax=np.pi, ymin=-8, ymax=8, s=1 ) );
text = ax_phase.text( 0, 3.5, f"t: {0}", fontsize=12 );

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

def update( val ):
    global t;
    f = f_slider.val;
    w = w_slider.val;
    t = 0;
    [ lines[ i ].set_ydata( angles_error[ :, i, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ] ) for i in range( len( lines ) ) ];
    poincare_recurrence_lines_x, poincare_recurrence_lines_y = poincare_recurrence( phase[ :, :, :, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ], error_phase[ :, :, :, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ] );
    for i in range( len( poincare_maps ) ):
        poincare_maps[ i ][ 0 ].set_offsets( np.column_stack( (poincare_recurrence_lines_x[ i ], poincare_recurrence_lines_y[ i ]) ) );
    for i in range( X ):
        poincare_graphs[ i ].set_offsets( np.column_stack( (poincare_x[ :, i, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ], poincare_y[ :, i, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ]) ) );
    fig.canvas.draw_idle();
    
t = 0;
dt_irl = 100.0 / 1000;
def Animate( frame ):
    global t;
    f = f_slider.val;
    w = w_slider.val;
    phase_space_graphs[ 0 ].set_offsets( np.column_stack( (angles[ find_nearest( time, t ), :, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ], angular_velocities[ find_nearest( time, t ), :, find_nearest( F_arr, f ), find_nearest( W_arr, w ) ]) ) );
    text.set_text( f"t: {t:.1f}" );
    t += dt_irl;
    if t > N * dt: t = 0;
    arr = phase_space_graphs;
    arr.append( text );
    return arr;

ani = animation.FuncAnimation( fig, Animate, frames=N, interval=dt_irl * 1000, blit=True );

f_slider.on_changed(update);
w_slider.on_changed(update);

plt.show();