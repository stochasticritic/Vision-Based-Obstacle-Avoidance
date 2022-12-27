import numpy as np
import scipy.optimize
#Implicit trajectory shape control variables
n_1 = 6
n_2 = 7

def snap_obj(x, *args):
    """
    For calculating the snap of the trajectory
    """
    snap = 0
    num_segments = (len(x) + n_1) // n_2
    for i in range(num_segments):
        idx = i * 7
        coeffs = x[idx:idx+7]
        snap += np.sum(coeffs[4:] ** 2)
    return snap

# Defining the starting position and the goal location
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([1, 1, 1, 0, 0, 0])

# Defining the waypoints array
waypoints = np.array([[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])

# Defining the total flight time including the waypont correspondance times
flight_time = 2.0
waypoint_times = np.array([0.5, 1.0])

# Traj. poly. degree, # of traj. segments
poly_degree = 3
num_segments = len(waypoints) + 1

# Opt. variables
num_coeffs = (poly_degree + 1) * num_segments
x0 = np.zeros(num_coeffs)

# Opt. constraints
constraints = []

# Initial state constraints
constraints.append({'type': 'eq', 'fun': lambda x: x[:7] - start})

# Final (goal) state constraints
constraints.append({'type': 'eq', 'fun': lambda x: x[-7:] - goal})

# Waypoint constraints
for i in range(num_segments - 1):
    t = waypoint_times[i]
    p = waypoints[i]
    idx = (poly_degree + 1) * (i + 1) - 1
    constraints.append({'type': 'eq', 'fun': lambda x: x[idx:idx+4] - p, 'args': (t,)})

# Minimizing the snap of the trajectory
result = scipy.optimize.minimize(snap_obj, x0, constraints=constraints)
coeffs = result.x

# Traj. Generation
t = np.linspace(0, flight_time, num=100)
trajectory = np.zeros((t.shape[0], 3))
for i in range(num_segments):
    # t_start and t_end for the current segment
    t_start = waypoint_times[i-1] if i > 0 else 0
    t_end = waypoint_times[i] if i < num_segments - 1 else flight_time
    
    # Generating poly. coeffs for each seg.
    idx = (poly_degree + 1) * i
    segment_coeffs = coeffs[idx:idx+poly_degree+1]

    # Genrating segment wise poly. functions
    segment_poly = np.poly1d(segment_coeffs)

    # Generating segment wise trajectories
    segment_t = t[(t >= t_start) & (t < t_end)]
    segment_trajectory = segment_poly(segment_t - t_start)

    # Completing the overall trajectory
    trajectory[(t >= t_start) & (t < t_end),:] = segment_trajectory

