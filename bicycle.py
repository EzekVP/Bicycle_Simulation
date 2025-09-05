import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

def distance_to_time_converter(distance_array: list[float], values_array: list[float], 
                              target_speed: float) -> tuple[list[float], list[float]]:
    """Convert distance-based profile to time-based profile"""
    time_array = [d / target_speed for d in distance_array]
    return time_array, values_array

def interpolate_profile(time_points: list[float], values: list[float], 
                       sim_time: list[float]) -> list[float]:
    """Interpolate profile values at simulation time points"""
    return np.interp(sim_time, time_points, values).tolist()

def calculate_gradient(elevation_profile: list[float], time_step: float, 
                      target_speed: float) -> list[float]:
    """Calculate road gradient from elevation profile"""
    dh_dt = np.gradient(elevation_profile, time_step)
    gradients = []
    for dh in dh_dt:
        theta = np.arctan(dh / target_speed)
        gradients.append(theta)
    return gradients

def bicycle_simulation(
    # Time domain
    time_array: list[float],
    
    # Input profiles (all in time domain)
    power_profile: list[float],           # W
    elevation_profile: list[float],       # m
    friction_profile: list[float],        # dimensionless
    wind_speed_profile: list[float],      # m/s
    wind_direction_profile: list[float],  # rad (0=tailwind, π=headwind)
    
    # Bicycle parameters
    total_mass: float,        # kg
    wheel_radius: float,      # m
    Crr: float,              # rolling resistance coefficient
    Cd: float,               # drag coefficient
    frontal_area: float,     # m²
    
    # Initial conditions
    initial_velocity: float,  # m/s
    initial_position: float = 0.0,  # m
    
    # Physical constants
    air_density: float = 1.225,  # kg/m³
    gravity: float = 9.81        # m/s²
) -> dict:
    
    # Initialize Gekko model
    m = GEKKO(remote=False)
    
    # Set time points
    m.time = time_array
    nt = len(time_array)
    dt = time_array[1] - time_array[0]
    
    # Calculate road gradients
    gradients = calculate_gradient(elevation_profile, dt, initial_velocity)
    
    # State variables
    v = m.Var(value=initial_velocity, name='velocity')  # velocity (m/s)
    x = m.Var(value=initial_position, name='position')  # position (m)
    
    # Input profiles as parameters
    P_given = m.Param(value=power_profile, name='power')
    mu_road = m.Param(value=friction_profile, name='friction')
    v_wind = m.Param(value=wind_speed_profile, name='wind_speed')
    wind_dir = m.Param(value=wind_direction_profile, name='wind_direction')
    theta = m.Param(value=gradients, name='gradient')
    
    # Intermediate calculations
    v_wind_component = m.Intermediate(-v_wind * m.cos(wind_dir), name='wind_component')
    v_relative = m.Intermediate(v + v_wind_component, name='relative_velocity')
    
    # Force components
    F_gravity = m.Intermediate(total_mass * gravity * m.sin(theta), name='gravity_force')
    F_rolling = m.Intermediate(Crr * mu_road * total_mass * gravity * m.cos(theta), name='rolling_force')
    
    # Aerodynamic drag (handle sign properly)
    F_aero = m.Intermediate(0.5 * air_density * Cd * frontal_area * v_relative * m.abs(v_relative), 
                           name='aero_force')
    
    # Driving force from power
    F_drive = m.Intermediate(P_given / v, name='driving_force')
    
    # Total resistance force
    F_resistance = m.Intermediate(F_gravity + F_rolling + F_aero, name='total_resistance')
    
    # Motion equations
    m.Equation(x.dt() == v)  # Position integration
    
    # Newton's second law: ma = F_net
    m.Equation(total_mass * v.dt() == F_drive - F_resistance)
    
    # Set solver options
    m.options.IMODE = 4  # Dynamic simulation
    m.options.NODES = 3
    m.options.SOLVER = 3
    
    # Solve
    m.solve(disp=False)
    
    # Extract results
    results = {
        'time': time_array,
        'velocity': v.value,
        'position': x.value,
        'power_given': power_profile,
        'elevation': elevation_profile,
        'gradient': gradients,
        'friction': friction_profile,
        'wind_speed': wind_speed_profile,
        'wind_direction': wind_direction_profile,
        'driving_force': [F_drive.value[i] for i in range(nt)],
        'gravity_force': [F_gravity.value[i] for i in range(nt)],
        'rolling_force': [F_rolling.value[i] for i in range(nt)],
        'aero_force': [F_aero.value[i] for i in range(nt)],
        'total_resistance': [F_resistance.value[i] for i in range(nt)]
    }
    
    return results

def n_phase_profile(
    time_sim: list[float],
    breakpoints: list[float],
    values: list[float],
    modes: list[str]
) -> list[float]:

    if len(breakpoints) != len(values):
        raise ValueError("breakpoints and values must have the same length")
    if len(modes) != len(breakpoints) - 1:
        raise ValueError("modes must have length len(breakpoints) - 1")

    prof: list[float] = []

    for t in time_sim:
        if t <= breakpoints[0]:
            i = 0
        elif t >= breakpoints[-1]:
            i = len(breakpoints) - 2
        else:
            i = max(j for j in range(len(breakpoints) - 1) if breakpoints[j] <= t)

        t0, t1 = breakpoints[i], breakpoints[i + 1]
        v0, v1 = values[i], values[i + 1]
        mode = modes[i]

        frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

        if mode == "constant":
            var = v1
        elif mode == "linear":
            var = v0 + (v1 - v0) * frac
        elif mode == "cosine":
            var = v0 + (v1 - v0) * (1 - np.cos(np.pi * frac)) / 2
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        prof.append(var)

    return prof

def friction_profile(
    time_sim: list[float],
    mode: str = "step",
    *,
    segments: list[tuple[float, float]] | None = None, 
    base: float = 0.005,
    amplitude: float = 0.002,
    frequency: float = 0.05,
    variation: float = 0.002,
    smooth_window: int = 1,
    low: float = 0.003,
    high: float = 0.01,
) -> list[float]:
    if mode == "step":
        if not segments:
            segments = [(0, base)]  # default flat asphalt
        values = []
        for t in time_sim:
            # pick last segment whose start_time <= t
            coeff = base
            for start, mu in segments:
                if t >= start:
                    coeff = mu
                else:
                    break
            values.append(coeff)
        return values

    if mode == "sinusoidal":
        return [
            base + amplitude * np.sin(frequency * t)
            for t in time_sim
        ]

    if mode == "random":
        raw = np.random.normal(base, variation, len(time_sim))
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            raw = np.convolve(raw, kernel, mode="same")
        return list(np.clip(raw, low, high))

    raise ValueError(f"Unknown mode '{mode}', choose from 'step', 'sinusoidal', 'random'")

def plot_simulation_results(results: dict) -> None:
    """Plot key simulation results"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))  # Changed to 2x4 grid for extra plots
    #fig.suptitle('Bicycle Simulation Results')
    
    time = results['time']

    # Velocity profile
    axes[0,0].plot(time, results['velocity'])
    axes[0,0].set_title('Velocity Profile')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Velocity (m/s)')
    axes[0,0].grid(True)

    # Power and forces
    axes[0,1].plot(time, results['power_given'], label='Given Power')
    axes[0,1].set_title('Power Profile')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Power (W)')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Forces breakdown
    axes[0,2].plot(time, results['driving_force'], label='Driving')
    axes[0,2].plot(time, results['gravity_force'], label='Gravity')
    axes[0,2].plot(time, results['rolling_force'], label='Rolling')
    axes[0,2].plot(time, results['aero_force'], label='Aero')
    axes[0,2].set_title('Force Components')
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Force (N)')
    axes[0,2].legend()
    axes[0,2].grid(True)

    # Friction vs Time
    axes[0,3].plot(time, results['friction'])
    axes[0,3].set_title('Friction vs Time')
    axes[0,3].set_xlabel('Time (s)')
    axes[0,3].set_ylabel('Friction Coefficient')
    axes[0,3].grid(True)

    # Elevation profile
    axes[1,0].plot(time, results['elevation'])
    axes[1,0].set_title('Elevation Profile')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Elevation (m)')
    axes[1,0].grid(True)

    # Gradient
    axes[1,1].plot(time, np.degrees(results['gradient']))
    axes[1,1].set_title('Road Gradient')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Gradient (degrees)')
    axes[1,1].grid(True)

    # Wind conditions
    axes[1,2].plot(time, results['wind_speed'], label='Wind Speed')
    axes[1,2].plot(time, np.degrees(results['wind_direction']), label='Wind Direction (deg)')
    axes[1,2].set_title('Wind Conditions')
    axes[1,2].set_xlabel('Time (s)')
    axes[1,2].set_ylabel('Speed (m/s) / Direction (deg)')
    axes[1,2].legend()
    axes[1,2].grid(True)

    # Friction vs Distance
    axes[1,3].plot(results['position'], results['friction'])
    axes[1,3].set_title('Friction vs Distance')
    axes[1,3].set_xlabel('Distance (m)')
    axes[1,3].set_ylabel('Friction Coefficient')
    axes[1,3].grid(True)

    plt.tight_layout()
    plt.show()

import csv
import os

def save_simulation_to_csv(results: dict, run_label: str, filename: str = "simulation_results.csv") -> None:
    """
    Save bicycle simulation results to CSV with a run label.

    Args:
        results: Dictionary returned by bicycle_simulation()
        run_label: String label for this simulation run
        filename: CSV filename to save results
    """
    # Ensure all lists have the same length
    n = len(results['time'])
    for key, val in results.items():
        if len(val) != n:
            raise ValueError(f"Length mismatch in results[{key}]")

    # Prepare CSV headers
    headers = [
        "run_label",
        "time_s",
        "position_m",
        "velocity_m_s",
        "power_W",
        "elevation_m",
        "gradient_rad",
        "friction",
        "wind_speed_m_s",
        "wind_dir_rad",
        "driving_force_N",
        "gravity_force_N",
        "rolling_force_N",
        "aero_force_N",
        "total_resistance_N"
    ]

    # Open CSV in append mode if file exists, otherwise write header
    write_header = not os.path.exists(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)

        # Write each time step
        for i in range(n):
            row = [
                run_label,
                results['time'][i],
                results['position'][i],
                results['velocity'][i],
                results['power_given'][i],
                results['elevation'][i],
                results['gradient'][i],
                results['friction'][i],
                results['wind_speed'][i],
                results['wind_direction'][i],
                results['driving_force'][i],
                results['gravity_force'][i],
                results['rolling_force'][i],
                results['aero_force'][i],
                results['total_resistance'][i]
            ]
            writer.writerow(row)

    print(f"Simulation results saved to {filename} with label: {run_label}")

# Example usage
if __name__ == "__main__":
    # Create sample profiles
    time_sim = list(np.linspace(0, 200, 101))  # 100 seconds, 1s resolution

    #Setting up profile ---> might be given to chatbot for user input/ or make direct changes to thissection to see change in o/p
    friction_control = "random" #Options: random, sinusoidal, step
    power = {'breakpoints': [0, 50, 100, 150, 200], 'power_values': [100, 100, 100, 100, 100], 'power_modes': ["cosine", "cosine", "cosine", "cosine"]}
    elevation = {'breakpoints': [0, 50, 100, 150, 200], 'elevation_values': [0, 10, 0, 10, 0], 'elevation_modes': ["cosine", "cosine", "cosine", "cosine"]}
    w_speed = {'breakpoints': [0, 50, 100, 150, 200], 'speed_values': [0, 8, 5, 5, 10], 'speed_modes': ["cosine", "cosine", "cosine", "cosine"]}
    w_dir = {'breakpoints': [0, 50, 100, 150, 200], 'dir_values': [0, np.pi/2, np.pi/2, np.pi, np.pi], 'dir_modes': ["cosine", "cosine", "cosine", "cosine"]}
     
    """Power profile""" 
    #Power profile: n-phase with breakpoints, values and modes
    power_prof = n_phase_profile(time_sim, breakpoints= power['breakpoints'], values=power['power_values'], modes=power['power_modes'])
   
    """Elevation profile"""
    #Elevation profile: n-phase with breakpoints, values and modes
    elevation_prof = n_phase_profile(time_sim, breakpoints= elevation['breakpoints'], values= elevation['elevation_values'], modes= elevation['elevation_modes'])

    """Friction profile"""
    if(friction_control == "sinusoidal"):
        friction_prof = friction_profile(time_sim, mode="sinusoidal", base=0.7, amplitude=0.1, frequency=0.03) #Sinusoidal variation

    elif(friction_control == "step"):
        friction_prof = friction_profile(time_sim, mode="step", segments=[(0, 0.7), (10, 0.6), (50, 0.8), (150, 0.9)])  # Step changes
    
    else:
        friction_prof = friction_profile(time_sim, mode="random", base=0.003, variation=5, smooth_window=5)

    """Wind profile"""
    # Wind-Speed profile: n-phase with breakpoints, values and modes
    wind_speed_prof = n_phase_profile(time_sim, breakpoints= w_speed['breakpoints'], values= w_speed['speed_values'], modes= w_speed['speed_modes'])


    # Wind-Direction profile: n-phase with breakpoints, values and modes
    wind_dir_prof = n_phase_profile(time_sim, breakpoints= w_dir['breakpoints'], values= w_dir['dir_values'], modes= w_dir['dir_modes'])
    
    # Bicycle parameters (typical road bike + rider)
    bike_params = {
        'total_mass': 80,      # kg (70kg rider + 10kg bike)
        'wheel_radius': 0.335,   # m (700c wheel)
        'Crr': 0.005,           # rolling resistance
        'Cd': 0.9,              # drag coefficient
        'frontal_area': 0.4,    # m² frontal area
    }
    
    # Run simulation
    results = bicycle_simulation(
        time_array=time_sim,
        power_profile=power_prof,
        elevation_profile=elevation_prof,
        friction_profile=friction_prof,
        wind_speed_profile=wind_speed_prof,
        wind_direction_profile=wind_dir_prof,
        initial_velocity=5.0,  # m/s (36 km/h)
        **bike_params
    )
    
    # Plot results
    plot_simulation_results(results)

    #Logging to CSV
    user_label = input("Enter a Label: ")
    run_label = "RUN 1: " + user_label
    save_simulation_to_csv(results, run_label)

    
    # Print summary
    final_velocity = results['velocity'][-1]
    avg_velocity = np.mean(results['velocity'])
    print(f"Initial velocity: {results['velocity'][0]:.2f} m/s")
    print(f"Final velocity: {final_velocity:.2f} m/s")
    print(f"Average velocity: {avg_velocity:.2f} m/s")

# bicycle_gekko_simulation.py (v2)------>Latest version with no assumptions might update a littel more