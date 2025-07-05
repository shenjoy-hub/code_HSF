import numpy as np

def smooth_step(x, x0=0, width=1, ymin=0, ymax=1, direction='up'):
    """
    Continuous approximation of a step function.
    
    Parameters:
    x (float or array): Input values
    x0 (float): Center of transition region
    width (float): Width of transition region
    ymin (float): Value left of transition
    ymax (float): Value right of transition
    direction (str): 'up' for increasing step, 'down' for decreasing step
    
    Returns:
    float or array: Output values
    
    The function uses a hyperbolic tangent (tanh) transition for smoothness.
    Transition region: [x0 - width/2, x0 + width/2]
    
    Example:
    >>> smooth_step(0.5)  # Basic step at 0
    0.5
    >>> smooth_step([-1, 0, 1], x0=0, width=1)
    [0.0, 0.5, 1.0]
    >>> smooth_step(2, x0=1, width=2, ymin=5, ymax=10)
    10.0
    """
    # Calculate normalized position in transition region
    position = (x - x0) / (width/2)
    
    # Calculate hyperbolic tangent transition (ranges from -1 to 1)
    transition = np.tanh(position * np.pi)  # π factor gives nice properties
    
    # Adjust transition to range [0, 1]
    normalized = (transition + 1) / 2
    
    # Apply direction handling
    if direction == 'down':
        normalized = 1 - normalized
        
    # Scale and translate to desired output range
    result = ymin + normalized * (ymax - ymin)
    
    return result

def step_function(tlist, step_times, values, side='right'):
    """
    Compute a strict step function (piecewise constant) at specified time points.
    
    Parameters:
        tlist (list or array): Times where to evaluate the step function
        step_times (list): Times where step changes occur (must be increasing)
        values (list): Function values between step points (length = len(step_times) + 1)
        side (str): 
            'right' - step changes at step_times (default)
            'left' - value changes after step_times
            'neither' - steps not included (exception if at step_time)
    
    Returns:
        ndarray: Step function values at tlist times
    
    Example:
        >>> t = [0.5, 1.0, 1.5, 2.0, 2.5]
        >>> step_times = [1.0, 2.0]
        >>> values = [0, 1, 0]
        >>> strict_step(t, step_times, values)
        array([0, 1, 1, 0, 0])   # For side='right'
        
        >>> strict_step(t, step_times, values, 'left')
        array([0, 0, 1, 1, 0])
    """
    # Ensure inputs are numpy arrays
    tlist = np.asarray(tlist)
    step_times = np.asarray(step_times)
    values = np.asarray(values)
    
    # Validate inputs
    if len(step_times) > 0:
        if not np.all(np.diff(step_times) > 0):
            raise ValueError("step_times must be strictly increasing")
            
    if len(values) != len(step_times) + 1:
        raise ValueError(f"values must have {len(step_times) + 1} elements")
    
    # Initialize result array with first value
    result = np.full(tlist.shape, values[0])
    
    # Handle all step times
    for i, step_time in enumerate(step_times):
        if side == 'right':
            mask = tlist >= step_time
        elif side == 'left':
            mask = tlist > step_time
        elif side == 'neither':
            mask = tlist > step_time
            # Skip the step_time points
            step_mask = tlist == step_time
            if np.any(step_mask):
                raise ValueError("tlist contains step_time point with side='neither'")
        else:
            raise ValueError("side must be 'left', 'right', or 'neither'")
            
        result[mask] = values[i + 1]
    
    return result

def smooth_square_wave(t, vmin, vmax, period, phase=0.0, n_terms=100):
    """
    Generate a smooth square wave oscillating between specified min and max values.
    
    Parameters:
        t (float or array): Time points to evaluate
        vmin (float): Minimum value of the wave
        vmax (float): Maximum value of the wave
        period (float): Period of oscillation
        phase (float): Phase shift in fractions of period (0 to 1) (default 0.0)
        n_terms (int): Number of harmonics to include (default 100)
        
    Returns:
        float or array: Square wave values at time points t
        
    Formula:
        f(t) = offset + amplitude * (4/π) * Σ [sin(2π(2m-1)(t/period + phase))/(2m-1)] 
        where:
            offset = (vmax + vmin)/2
            amplitude = (vmax - vmin)/2
            
    Example:
        >>> t = np.linspace(0, 2, 1000)
        >>> y = smooth_square_wave(t, vmin=0, vmax=5, period=1.0)
    """
    t = np.asarray(t)
    
    # Calculate offset and amplitude
    offset = (vmax + vmin) / 2
    amplitude = (vmax - vmin) / 2
    
    # Angular frequency
    ω = 2 * np.pi / period
    
    # Initialize result
    result = np.zeros_like(t)
    
    # Sum Fourier series
    for m in range(1, n_terms + 1):
        n = 2 * m - 1  # Odd harmonics only
        result += np.sin(n * ω * (t + phase * period)) / n
    
    # Apply scaling and offset
    result *= 4 * amplitude / np.pi
    result += offset
    
    return result

# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create data
    x = np.linspace(-2, 2, 1000)
    
    # Different parameter configurations
    y1 = smooth_step(x, x0=0, width=1)            # Basic step
    y2 = smooth_step(x, x0=0, width=0.01)          # Steeper transition
    y3 = smooth_step(x, x0=0.5, width=1)           # Shifted center
    y4 = smooth_step(x, x0=-1, width=1, ymin=2, ymax=5)  # Different range
    y5 = smooth_step(x, x0=0, width=1, direction='down') # Down step
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y1, 'b-', linewidth=2, label='Basic: center=0, width=1')
    plt.plot(x, y2, 'r--', linewidth=2, label='Steep: width=0.5')
    plt.plot(x, y3, 'g-.', linewidth=2, label='Shifted: center=0.5')
    plt.plot(x, y4, 'm:', linewidth=2, label='Scaled: min=2, max=5')
    plt.plot(x, y5, 'c-', linewidth=2, label='Down: direction="down"')
    
    # Styling
    plt.title("Smooth Step Functions", fontsize=14)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()