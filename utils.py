import numpy as np
import matplotlib.pyplot as plt

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
    result = 0.0
    
    # Sum Fourier series
    for m in range(1, n_terms + 1):
        n = 2 * m - 1  # Odd harmonics only
        result += np.sin(n * ω * (t + phase * period)) / n
    
    # Apply scaling and offset
    result *= 4 * amplitude / np.pi
    result += offset
    
    return result

def fourier_transform(x, y, do_plot=True):
    """
    Compute and optionally plot the Fourier transform of a signal.
    
    Parameters:
        x (array_like): Time domain values (uniformly sampled)
        y (array_like): Signal amplitudes corresponding to time values
        do_plot (bool): If True, generates time-domain and frequency-domain plots
    
    Returns:
        freqs (ndarray): Frequency values (two-sided or one-sided)
        magnitude (ndarray): Magnitude of Fourier transform components
    """
    # Validate inputs
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    # Calculate time step and check uniformity
    dt = np.mean(np.diff(x))
    if not np.allclose(np.diff(x), dt, rtol=1e-5, atol=1e-8):
        raise ValueError("Time samples must be uniformly spaced")
    
    N = len(y)  # Number of samples
    
    # Compute FFT and normalize magnitudes
    fft_vals = np.fft.fft(y)
    magnitude = np.abs(fft_vals) / N  # Normalized magnitude
    
    # Frequency calculations
    freqs = np.fft.fftfreq(N, dt)  # Two-sided frequencies
    
    # Create one-sided representation
    pos_idxs = np.where(freqs >= 0)  # Positive frequency indices
    freqs_one = freqs[pos_idxs]
    mag_one = magnitude[pos_idxs].copy()
    
    # Normalize one-sided magnitudes (double for energy conservation)
    if N % 2 == 0:  # Even number of samples
        mag_one[1:-1] *= 2  # Exclude DC (0) and Nyquist frequency
    else:  # Odd number of samples
        mag_one[1:] *= 2
    
    # Generate plots if requested
    if do_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Time domain plot
        ax1.plot(x, y)
        ax1.set_title('Time Domain Signal')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Frequency domain plot (one-sided)
        ax2.plot(freqs_one, mag_one)
        ax2.set_title('Frequency Domain (One-Sided)')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True)
        ax2.set_xlim(left=0)
        
        plt.tight_layout()
        plt.show()
    
    return freqs_one, mag_one

# Example usage:
if __name__ == "__main__":
    # Generate sample signal
    fs = 500# Sampling frequency
    t = np.arange(0, 2, 1/fs)  # Time vector (1 second duration)
    signal = 3 * np.cos(2*np.pi*50*t) + 2 * np.sin(2*np.pi*120*t)-3+ 3 * np.cos(2*np.pi*51*t)
    
    # Call function with plotting
    freqs, mag = fourier_transform(t, signal, do_plot=True)