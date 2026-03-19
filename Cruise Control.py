import matplotlib.pyplot as plt
from random import random
import numpy as np

class VehicleMotion:
    #Simulates motion of vehicle as time progresses
    def __init__(self, mass: float = 1500.0, drag_coeff: float = 0.3, cross_area: float = 2.2, air_density: float = 1.225, 
                 rolling_fric_coeff: float = 0.01):
        self.mass = mass
        self.drag_const = 0.5 * drag_coeff * cross_area * air_density
        self.F_fric = rolling_fric_coeff * mass * 9.81
    
    def step(self, current_velocity: float, F_thrust: float, dt: float) -> tuple[float, float]:
        F_drag = self.drag_const * current_velocity**2
        F_net = F_thrust - F_drag - self.F_fric
        acceleration = F_net/self.mass
        new_velocity = current_velocity + acceleration * dt
        return new_velocity, acceleration

class PIDController:
    #PID controller with asymmetrical integral application (preventing windup)
    def __init__(self, kp: float, ki: float, kd: float, windup_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        self.total_error = 0
        self.prev_error = None
        self.d_last = 0
        self.alpha = 0.1
    
    def PID(self, target: float, current: float, dt: float, min_thrust: float, max_thrust: float) -> float:
        current_error = target - current
        if self.prev_error == None:
            self.prev_error = current_error
        #Anti-Windup clamp, fixes total_error above 0 to minimise overshoot from above caused by drag forces
        if abs(current_error) < 2.0:
            self.total_error += current_error
            #Asymmetrical approach helps overshoot caused by opposing drag and rolling friction forces
            self.total_error = max(0, min(self.total_error, self.windup_limit))

        p = self.kp * current_error
        i = self.ki * self.total_error * dt
        raw_d = self.kd * ((current_error-self.prev_error)/dt)
        d = (self.alpha * raw_d) + ((1-self.alpha) * self.d_last)

        self.prev_error = current_error
        self.d_last = d

        #Constrains max and min engine thrust force.
        return max(min_thrust, min(p + i + d, max_thrust))

class KalmanFilter:
    #Filters randomised data using Kalman filter, applied with numpy arrays
    def __init__(self, dt: float, gps_var: float, vel_var: float, weight: float):
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.5 * dt**2], [dt]])
        self.H = np.identity(2)
        self.R = np.array([[gps_var, 0.0], [0.0, vel_var]])
        self.Q = weight * np.identity(2)
        self.P = np.identity(2)
        self.I = np.identity(2)
        self.state_estimate = np.zeros((2, 1))
    def filter(self, state_measured: np.ndarray, acceleration: float):
        #Predicting the measured state
        self.state_estimate = self.A @ self.state_estimate + self.B * acceleration
        self.P = self.A @ self.P @ np.transpose(self.A) + self.Q

        #Update state estimate and estimate covariance (uncertainty of guess made by filter)
        estimate_error = state_measured - self.state_estimate
        self.K = self.P @ np.transpose(self.H) @ np.linalg.inv(self.H @ self.P @ np.transpose(self.H) + self.R)
        self.state_estimate = self.state_estimate + self.K @ estimate_error
        self.P = (self.I - self.K @ self.H) @ self.P
        return self.state_estimate

def graph_results(real, kalman, noisy, thrust, target):
    #Plot two graphs, one for velocity and one for thrust
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    #Top Plot - Velocity and filtering
    ax1.plot(noisy, color='lightgray', alpha=0.7, label='Raw Sensor (Garbage Data)')
    ax1.plot(kalman, color='orange', linewidth=2, label='Kalman Filter')
    ax1.plot(real, color='blue', linewidth=2, label='True Velocity')
    ax1.axhline(y=target, color='red', linestyle='--', label='Target Velocity')
    ax1.set_title("System State: Velocity vs Target")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    #Bottom Plot - Thrust Force
    ax2.plot(thrust, color='purple', linewidth=2, label='Engine Thrust (PID Output)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1) # Baseline zero thrust
    ax2.set_title("Control Effort: Engine Force Applied")
    ax2.set_xlabel("Time Steps (dt = 0.1s)")
    ax2.set_ylabel("Force (Newtons)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_simulation():
    #Simulation constants
    initial_velocity = 20.0 #m/s
    time_step = 0.1 #s
    target_velocity = 30 #m/s
    steps = 500
    engine_strength = [-10000, 10000]

    #Filtering constants
    rand_gps = 10 #Range of randomisation being applied to current location
    rand_vel = 1 #Range of randomisation being applied to current velocity
    measured_estimate_weight = 0.001 #Weight given to measured values for location and velocity over predicted ones (keep low)

    #Initialise variables
    real_pos = 0.0
    real_vel = initial_velocity
    accel = 0.0

    car = VehicleMotion()
    pid = PIDController(kp = 1000.0, ki = 50.0, kd = 100.0, windup_limit = (target_velocity ** 2))
    kalman = KalmanFilter(dt = time_step, gps_var = (rand_gps**2)/12, vel_var = (rand_vel**2)/12, weight = measured_estimate_weight)
    kalman.state_estimate = np.array([[real_pos], [initial_velocity]])

    history_real, history_noisy, history_kalman, history_thrust = [], [], [], []

    for _ in range(steps):
        #Simulate real-life measurements by adding inaccuracy
        noisy_motion = np.array([
            [real_pos + (rand_gps * random()) - (rand_gps / 2.0)], 
            [real_vel + (rand_vel * random()) - (rand_vel / 2.0)]
        ])

        #Estimate motion
        estimate_motion = kalman.filter(noisy_motion, accel)
        estimate_vel = estimate_motion[1][0]

        #Calculate thrust with PID controller
        F_thrust = pid.PID(target_velocity, estimate_vel, time_step, engine_strength[0], engine_strength[1])

        #Apply physics
        real_pos += real_vel * time_step
        real_vel, accel = car.step(real_vel, F_thrust, time_step)

        #Log data
        history_real.append(real_vel)
        history_noisy.append(noisy_motion[1][0])
        history_kalman.append(estimate_vel)
        history_thrust.append(F_thrust)

    graph_results(history_real, history_kalman, history_noisy, history_thrust, target_velocity)

if __name__ == "__main__":
    run_simulation()