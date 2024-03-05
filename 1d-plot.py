import matplotlib.pyplot as plt;
import numpy as np;

    
def plot_multiple_lines(xdata, ydata):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(len(ydata[0])):
        ax.plot(xdata, ydata[:,i])
    
    ax.set_xlabel('time')
    ax.set_ylabel('x')
    ax.set_title('Initial Condition Sensitivity Plot')
    
    plt.tight_layout()
    plt.show()

def TimeDerivative( x ):
    return np.cos( x );

def Iterate( init_cond, N, dt, X ):
    f = np.zeros( (N,X) );
    f[ 0 ] = init_cond;
    for i in range( 1, N ):
        f[ i ] = f[ i - 1 ] + TimeDerivative( f[ i - 1 ] ) * dt;
    return f;

N = 100;
dt = 0.1;
X = 100;
x = np.linspace( 9.9, 10.1, X );
y = Iterate( x, N=N, dt=0.1, X=X );

plot_multiple_lines( np.linspace( 0, N * dt, N ), y );