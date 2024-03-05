import matplotlib.pyplot as plt;
import numpy as np;

    
def plot_multiple_lines(xdata, ydata):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(len(ydata[0])):
        ax.plot(xdata, ydata[:,i])
    
    ax.set_xlabel('n (iterations)')
    ax.set_ylabel('Y')
    ax.set_title('Multiple Lines Plot')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def IteratedFunc( x ):
    return np.tan( x );

def Iterate( init_cond, N, X ):
    f = np.zeros( (N,X) );
    f[ 0 ] = init_cond;
    for i in range( 1, N ):
        f[ i ] = IteratedFunc( f[ i - 1 ] );
    return f;

N = 100;
X = 100;
x = np.linspace( 9.9, 10.1, X );
y = Iterate( x, N=N, X=X );

plot_multiple_lines( np.arange( N ), y );