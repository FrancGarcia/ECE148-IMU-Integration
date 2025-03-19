import logging as logger
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

import ekf_imu

class ArtemisOpenLog:
    # Constructor
    def __init__(self, port, baudrate, time):
        assert(isinstance(port, str)), "Port must be a valid string input"
        assert(isinstance(baudrate, int) and baudrate > 0), "Baudrate must be valid int"
        assert((isinstance(time, int) or isinstance(time, float)) and time > 0), "timeout time must be valid number greater than 0" 
        self.ser = None
        self.accel = { 'x': 0, 'y': 0, 'z': 0}
        self.gyro = { 'x' : 0, 'y': 0, 'z': 0}
        self.gyro_smooth = { 'x' : 0, 'y': 0, 'z': 0}
        self.euler = { 'x': 0, 'y': 0, 'z': 0}
        self.mag = { 'x': 0, 'y': 0, 'z': 0}
        # Accel data is in milli g
        self.accel_x = deque(maxlen=100)
        self.accel_y = deque(maxlen=100)
        self.accel_z = deque(maxlen=100)
        # Gyroscope data is in degrees per second
        self.gyro_x_s = deque(maxlen=100)
        self.gyro_y_s = deque(maxlen=100)
        self.gyro_z_s = deque(maxlen=100)
        self.gyro_x = deque(maxlen=100)
        self.gyro_y = deque(maxlen=100)
        self.gyro_z = deque(maxlen=100)
        # Euler data is in degrees per second
        self.euler_x = deque(maxlen=100)
        self.euler_y = deque(maxlen=100)
        self.euler_z = deque(maxlen=100)
        try:
            self.ser = serial.Serial(port, baudrate, timeout=time)  # Fix: pass timeout as a keyword argument
            logger.info(f"Connected to {port}")
        except serial.SerialException as e:
            logger.error(f"Error: {e}")
        except KeyboardInterrupt:
            logger.info("\nExiting...")

    def poll(self):
        """
        Reads the IMU data. Sets the Artemis IMU object fields to the data readings.
        """
        self.ser.reset_input_buffer()        
        data = self.ser.readline().decode('utf-8').strip().split(",")
        try:
            self.accel = { 'x' : float(data[2]) * 0.00981, 'y' : float(data[3]) * 0.00981, 'z' : float(data[4]) * 0.00981 }
            self.gyro = { 'x': float(data[5]), 'y': float(data[6]), 'z': float(data[7]) }
            self.mag = { 'x': float(data[8]), 'y': float(data[9]), 'z': float(data[10]) }

            # Noisy gyroscope data
            self.gyro_x.append(self.gyro['x'])
            self.gyro_y.append(self.gyro['y'])
            self.gyro_z.append(self.gyro['z'])

            # Smooth gyroscope data
            x = np.zeros(3) 
            P = np.eye(3)  
            Q = np.eye(3) * 0.001  
            R = np.eye(3) * 10 
            z = np.array([self.gyro['x'], self.gyro['y'], self.gyro['z']])
            x, P = self.kalman_filter(x, P, z, Q, R)
            self.gyro_smooth = { 'x': float(x[0]), 'y': float(x[1]), 'z': float(x[2]) }
            self.gyro_x_s.append(self.gyro_smooth['x'])
            self.gyro_y_s.append(self.gyro_smooth['y'])
            self.gyro_z_s.append(self.gyro_smooth['z'])

            # Get euler angles
            ekf = ekf_imu.EKF_IMU(dt=0.01)
            g = np.array([v for v in self.gyro_smooth.values()])
            ekf.predict(g)
            a = np.array([v for v in self.accel.values()])
            m = np.array([v for v in self.mag.values()])
            ekf.update(a, m) 
            euls = ekf.get_euler_angles()
            self.euler = { 'x' : float(euls[0]), 'y': float(euls[1]), 'z': float(euls[2])}
            self.euler_x.append(self.euler['x'])
            self.euler_y.append(self.euler['y'])
            self.euler_z.append(self.euler['z'])
            
        except IndexError:
            logger.error("Waiting for IMU data to parse")
        except ValueError:
            logger.error("Waiting for IMU data to cast to floats")
    
    def kalman_filter(self, x, P, z, Q, R):
        """
        Applies 3D Kalman Filter to smoothen the data. Must smoothen
        the IMU data to get better sensor fusion with the GPS.

        :param x: The Initial state that we want to predict (3x1 vector)
        :param P: The initial uncertainty covariance of our initial data readings (3x3 matrix)
        :param z: The noisy data to smoothen out (3x1 vector)
        :param Q: Process noise covariance (3x3 matrix)
        :param R: Measurement noise covariance (3x3 matrix)

        :return: Updated/corrected data and covariance uncertainty (x_updated, P_updated)
        """
        assert(isinstance(x, np.ndarray) and x.shape == (3,)), "x must be a 3x1 numpy array"
        assert(isinstance(P, np.ndarray) and P.shape == (3, 3)), "P must be a 3x3 numpy array"
        assert(isinstance(z, np.ndarray) and z.shape == (3,)), "z must be a 3x1 numpy array"
        assert(isinstance(Q, np.ndarray) and Q.shape == (3, 3)), "Q must be a 3x3 numpy array"
        assert(isinstance(R, np.ndarray) and R.shape == (3, 3)), "R must be a 3x3 numpy array"

        # Predict the new data
        F = np.eye(3)
        x_pred = np.dot(F, x)
        P_pred = np.dot(np.dot(F,P), F.T) + Q

        # Update/Correct the raw data
        H = np.eye(3)
        y_residual = z - np.dot(H, x_pred)
        S = np.dot(np.dot(H, P_pred), H.T) + R
        K_gain = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
        x_updated = x_pred + np.dot(K_gain, y_residual)
        P_updated = P_pred - np.dot(np.dot(K_gain, H), P_pred)

        return x_updated, P_updated

    def run(self):
        """
        Calls the poll() method to read the IMU data.

        :return: dict of floats, dict of floats
            Gyroscope readings as a dictionary, Acceleration readings as a dictionary
        """
        self.poll()
        return self.euler, self.accel, self.gyro_smooth

    def run_threaded(self):
        """
        Returns the gyroscope readings and the acceleration readings in a thread.

        :return: dict of floats, dict of floats
            Gyroscope readings as a dictionary, Acceleration readings as a dictionary
        """
        return self.euler, self.accel, self.gyro

    def shutdown(self):
        """
        Closes the serial communication with the Artemis IMU.
        """
        if self.ser:
            self.ser.close()
            logger.info("Closed serial connection")

# ---------- Free Functions ---------- #

def log_data(artemis_imu: ArtemisOpenLog):
    """
    Plots the data in real time.

    :param: artemis_imu
        The ArtemisOpenLog object to use to collect the data.
    """
    try:
        def update_plot(_):
            artemis_imu.poll()
            # line_gx.set_data(range(len(artemis_imu.gyro_x)), list(artemis_imu.gyro_x))
            # line_gy.set_data(range(len(artemis_imu.gyro_y)), list(artemis_imu.gyro_y))
            line_gz.set_data(range(len(artemis_imu.gyro_z)), list(artemis_imu.gyro_z))

            # line_gx_s.set_data(range(len(artemis_imu.gyro_x_s)), list(artemis_imu.gyro_x_s))
            # line_gy_s.set_data(range(len(artemis_imu.gyro_y_s)), list(artemis_imu.gyro_y_s))
            line_gz_s.set_data(range(len(artemis_imu.gyro_z_s)), list(artemis_imu.gyro_z_s))

            ax.set_xlim(0, len(artemis_imu.gyro_z))
            # return line_gx, line_gy, line_gz, line_gx_s, line_gy_s, line_gz_s
            return line_gz, line_gz_s

        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
        ax.set_ylabel("Gyroscope (deg/s)")
        ax.set_ylim(-10, 10)
        # line_gx, = ax.plot([], [], label="gX", color="r")
        # line_gy, = ax.plot([], [], label="gY", color="g")
        line_gz, = ax.plot([], [], label="gZ", color="b")

        # line_gx_s, = ax.plot([], [], label="gX_s", color="orange")
        # line_gy_s, = ax.plot([], [], label="gY_s", color="purple")
        line_gz_s, = ax.plot([], [], label="gZ_s", color="red")        
        ax.legend()

        ani = animation.FuncAnimation(fig, update_plot, interval=5, blit=True, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        logger.info("\nStopping the plotting...")

def log_data_accel(artemis_imu: ArtemisOpenLog):
    """
    Plots the data in real time.

    :param: artemis_imu
        The ArtemisOpenLog object to use to collect the data.
    """
    try:
        def update_plot(_):
            artemis_imu.poll()
            # line_gx.set_data(range(len(artemis_imu.gyro_x)), list(artemis_imu.gyro_x))
            # line_gy.set_data(range(len(artemis_imu.gyro_y)), list(artemis_imu.gyro_y))
            line_az.set_data(range(len(artemis_imu.accel_z)), list(artemis_imu.accel_z))

            # line_gx_s.set_data(range(len(artemis_imu.gyro_x_s)), list(artemis_imu.gyro_x_s))
            # line_gy_s.set_data(range(len(artemis_imu.gyro_y_s)), list(artemis_imu.gyro_y_s))
            line_az_s.set_data(range(len(artemis_imu.accel_z_s)), list(artemis_imu.accel_z_s))

            ax.set_xlim(0, len(artemis_imu.accel_z))
            # return line_gx, line_gy, line_gz, line_gx_s, line_gy_s, line_gz_s
            return line_az, line_az_s

        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
        ax.set_ylabel("Acceleration (meters/s^2)")
        ax.set_ylim(-10, 10)
        # line_gx, = ax.plot([], [], label="gX", color="r")
        # line_gy, = ax.plot([], [], label="gY", color="g")
        line_az, = ax.plot([], [], label="aZ", color="b")

        # line_gx_s, = ax.plot([], [], label="gX_s", color="orange")
        # line_gy_s, = ax.plot([], [], label="gY_s", color="purple")
        line_az_s, = ax.plot([], [], label="aZ_s", color="red")        
        ax.legend()

        ani = animation.FuncAnimation(fig, update_plot, interval=5, blit=True, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        logger.info("\nStopping the plotting...")

def log_data_euler(artemis_imu: ArtemisOpenLog):
    """
    Plots the data in real time.

    :param: artemis_imu
        The ArtemisOpenLog object to use to collect the data.
    """
    try:
        def update_plot(_):
            artemis_imu.poll()
            # line_gx.set_data(range(len(artemis_imu.gyro_x)), list(artemis_imu.gyro_x))
            # line_gy.set_data(range(len(artemis_imu.gyro_y)), list(artemis_imu.gyro_y))
            line_ez.set_data(range(len(artemis_imu.euler_z)), list(artemis_imu.euler_z))

            ax.set_xlim(0, len(artemis_imu.euler_z))
            return line_ez

        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
        ax.set_ylabel("Euler (degrees)")
        ax.set_ylim(-10, 10)
        # line_gx, = ax.plot([], [], label="gX", color="r")
        # line_gy, = ax.plot([], [], label="gY", color="g")
        line_ez, = ax.plot([], [], label="eulerZ", color="b")     
        ax.legend()

        ani = animation.FuncAnimation(fig, update_plot, interval=5, blit=False, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        logger.info("\nStopping the plotting...")


if __name__ == "__main__":

    #SERIAL_PORT = "COM9" # Used Windows OS
    SERIAL_PORT = "/dev/ttyUSB0" # Used for Jetson Nano 
    artemis_imu = ArtemisOpenLog(SERIAL_PORT, 115200, 1)

    # For plotting the gyroscope data
    # log_data(artemis_imu)
    # log_data_accel(artemis_imu)
    # log_data_euler(artemis_imu)
    # artemis_imu.shutdown()
    
    try:
        while True:
            euler, accel, gyro = artemis_imu.run()
            print(f"Gyro: {gyro}, Accel: {accel}")
            #print(f"Eulers: {euler}")
            time.sleep(0.1)
    except KeyboardInterrupt:
         logger.info("\nShutting down...")
         artemis_imu.shutdown()

