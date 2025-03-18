import serial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from collections import deque
import time
import logging as logger
import threading
import csv
import numpy as np
from scipy.signal import butter, filtfilt

class ArtemisOpenLog:
    def __init__(self, port, baudrate, timeout):
        assert(isinstance(port, str)), "Port must be a valid string input"
        assert(isinstance(baudrate, int) and baudrate > 0), "Baudrate must be valid int"
        assert((isinstance(timeout, int) or isinstance(timeout, float)) and timeout > 0), "timeout time must be valid number greater than 0" 
        self.serial_port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.running = False
        self.read_imu_thread = None

        self.gyro_offset = None
        self.accel_offset = None

        # Acceleration is in milli g
        self.accel = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.accel_x = deque(maxlen=100)
        self.accel_y = deque(maxlen=100)
        self.accel_z = deque(maxlen=100)
        # Gyroscope data is in degrees per second
        self.gyro = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.gyro_x = deque(maxlen=100)
        self.gyro_y = deque(maxlen=100)
        self.gyro_z = deque(maxlen=100)
        self.temp = 0.0

        self.yaw = 0.0
        self.yaw_values = deque(maxlen=100)  # Deque to store yaw values
        self.ekf_state = np.zeros(2)  # [yaw, yaw_rate]
        self.ekf_covariance = np.eye(2)  # Initial covariance matrix

        self.connect()
    
    def connect(self):
        """
        Sets up the serial communication of the OpenLog IMU
        via UART with the given port, baudrate, and the timeout.

        :param port: The serial port to communicate through
        :param baudrate: The baudrate of the serial communcation
        :param timeout: The timeout
        """
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=self.timeout)
            logger.info(f"Connected to {self.serial_port}")
        except serial.SerialException as e:
            logger.error(f"Error: {e}")
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            self.shutdown()
            

    def read_imu_data(self, log_data: bool = None, log_type: str = None):
        """
        Read the data from the ICM20948 IMU
        that is on the Artemis OpenLog.

        :param log_data: Boolean to determine if we want to also log the data in matplotlib figure.
        :param log_type: String to determine what to log on the matplotlib figure if we want to plot.
        """
        assert(isinstance(log_data, bool) or log_data is None), "log_data must be True or False"
        assert((isinstance(log_type, str) or log_type is None) and (log_type == "accel" or log_type == "gyro" or log_type is None)), "log_type must be 'accel' or 'gyro'"

        if self.ser is None:
            logger.error("Serial connection not initialized")
            return

        while self.running:

            #with open('imu_data_threaded.csv', 'w', newline='') as fp_imu:
                
            try:
                # Artemis OpenLog already records data in CSV format for us
                # Look at datasheet for Artemis OpenLog Sparkfun to see what data
                # points are stored in which indices in the data array when reading 
                data = self.ser.readline().decode('utf-8').strip().split(",")
                str_vals = ['rtcDate', 'rtcTime', 'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ', 'mX', 'mY', 'mZ', 'imu_degC', 'output_Hz']
                if len(data) >= 12:

                    data_array = data[0:len(data)-1]
                    
                    print(data_array)

                    # This if statement is critical for plotting the data in real time.
                    if data_array != str_vals:
                        # Apply accel offset if available
                        if self.accel_offset:
                            self.accel = {
                                'x' : float(data_array[2]) * 0.00981 - self.accel_offset['x'],
                                'y' : float(data_array[3]) * 0.00981 - self.accel_offset['y'],
                                'z' : float(data_array[4]) * 0.00981 - self.accel_offset['z']
                            }
                        else:
                            self.accel = {
                                'x' : float(data_array[2]) * 0.00981,
                                'y' : float(data_array[3]) * 0.00981,
                                'z' : float(data_array[4]) * 0.00981
                            }
                        # For logging the previous 100 points if necessary
                        self.accel_x.append(self.accel['x'])
                        self.accel_y.append(self.accel['y'])
                        self.accel_z.append(self.accel['z'])
                        
                        # Apply gyro offset if available
                        if self.gyro_offset:
                            raw_gyro = { 
                                'x' : float(data_array[5]) - self.gyro_offset['x'], 
                                'y' : float(data_array[6]) - self.gyro_offset['y'], 
                                'z' : float(data_array[7]) - self.gyro_offset['z'] 
                            }
                        else:
                            raw_gyro = { 
                                'x' : float(data_array[5]), 
                                'y' : float(data_array[6]), 
                                'z' : float(data_array[7]) 
                            }
                        # For logging the previous 100 points if necessary
                        self.gyro_x.append(self.gyro['x'])
                        self.gyro_y.append(self.gyro['y'])
                        self.gyro_z.append(self.gyro['z'])
                        self.temp = float(data_array[11])


                        # Kalman Filter parameters to smoothen the gyroscope data
                        # Depending on your system, may need to tune Q and R by trial-and-error
                        # Found the below parameters to work really well
                        x = np.zeros(3) 
                        P = np.eye(3)  
                        Q = np.eye(3) * 0.001  
                        R = np.eye(3) * 10 
                        z = np.array([raw_gyro['x'], raw_gyro['y'], raw_gyro['z']])
                        x, P = self.kalman_filter_gyro(x, P, z, Q, R)

                        self.gyro = {
                            'x': x[0],
                            'y': x[1],
                            'z': x[2]
                        }

                        self.gyro_x.append(self.gyro['x'])
                        self.gyro_y.append(self.gyro['y'])
                        self.gyro_z.append(self.gyro['z'])
                        self.temp = float(data_array[11])

                        dt = 0.01  # Assuming a fixed time step for simplicity
                        mag_yaw = np.arctan2(float(data_array[9]), float(data_array[8]))
                        # Use the EKF sensor fusion between gyroscope
                        # and magnometer to get the yaw angle 
                        self.ekf_gyro_mag(dt, mag_yaw)
                        self.yaw = self.ekf_state[0]
                        self.yaw_values.append(self.yaw)

            except Exception as e:
                logger.error(f"Error reading data from Artemis IMU: {e}")
                time.sleep(0.1)

    def start_imu_threading(self):
        """
        Starts the thread for reading the IMU data to
        run in the backround while also being connected.
        """
        if self.running and self.read_imu_thread is not None:
            logger.info("IMU data reading thread is already running")
            return
        if self.read_imu_thread is None or not self.read_imu_thread.is_alive():
            self.running = True
            self.read_imu_thread = threading.Thread(target=self.read_imu_data, daemon=True)
            self.read_imu_thread.start()
            logger.info("Started IMU data reading thread")

    def stop_imu_threading(self):
        """
        Stops the thread for reading the IMU data.
        """
        self.running = False
        if self.read_imu_thread:
            self.read_imu_thread.join()
            logger.info("Stopped IMU data readingt thread")

    def ekf_gyro_mag(self, dt, mag_yaw):
        """
        Extended Kalman Filter to predict and update the state based on gyroscope and magnetometer data.

        :param dt: Time step between predictions.
        :param mag_yaw: Yaw angle from the magnetometer.
        """
        assert((isinstance(dt, int) or isinstance(dt, float)) and dt > 0), "dt must be a valid number for EKF"
        assert(isinstance(mag_yaw, float)), "The yaw angle must be a valid float"

        # Prediction step
        F = np.array([[1, dt], [0, 1]]) 
        Q = np.array([[0.01, 0], [0, 0.01]]) 

        self.ekf_state = np.dot(F, self.ekf_state)
        self.ekf_covariance = np.dot(np.dot(F, self.ekf_covariance), F.T) + Q

        # Update/Correct the yaw angle 
        H = np.array([[1, 0]])
        R = np.array([[0.1]]) 

        y = mag_yaw - np.dot(H, self.ekf_state)
        S = np.dot(np.dot(H, self.ekf_covariance), H.T) + R 
        K = np.dot(np.dot(self.ekf_covariance, H.T), np.linalg.inv(S))

        self.ekf_state = self.ekf_state + np.dot(K, y)
        self.ekf_covariance = np.dot((np.eye(2) - np.dot(K, H)), self.ekf_covariance)

    def kalman_filter_gyro(self, x, P, z, Q, R):
        """
        Applies 3D Kalman Filter to smoothen the gyroscope data. Must smoothen
        the gyroscope data to get better sensor fusion with the GPS.

        :param x: The Initial state that we want to predict (3x1 vector)
        :param P: The initial uncertainty covariance of our initial data readings (3x3 matrix)
        :param z: The noisy gyroscope data to smoothen out (3x1 vector)
        :param Q: Process noise covariance (3x3 matrix)
        :param R: Measurement noise covariance (3x3 matrix)

        :return: Updated/corrected gyroscope data and covariance uncertainty (x_updated, P_updated)
        """
        assert(isinstance(x, np.ndarray) and x.shape == (3,)), "x must be a 3x1 numpy array"
        assert(isinstance(P, np.ndarray) and P.shape == (3, 3)), "P must be a 3x3 numpy array"
        assert(isinstance(z, np.ndarray) and z.shape == (3,)), "z must be a 3x1 numpy array"
        assert(isinstance(Q, np.ndarray) and Q.shape == (3, 3)), "Q must be a 3x3 numpy array"
        assert(isinstance(R, np.ndarray) and R.shape == (3, 3)), "R must be a 3x3 numpy array"

        # Predict the new gyroscope data
        F = np.eye(3)
        x_pred = np.dot(F, x)
        P_pred = np.dot(np.dot(F,P), F.T) + Q

        # Update/Correct the raw gyroscope data
        H = np.eye(3)
        y_residual = z - np.dot(H, x_pred)
        S = np.dot(np.dot(H, P_pred), H.T) + R
        K_gain = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
        x_updated = x_pred + np.dot(K_gain, y_residual)
        P_updated = P_pred - np.dot(np.dot(K_gain, H), P_pred)

        return x_updated, P_updated

    def low_pass_filter_gyro(self, cutoff, freq, data, order=5):
        """
        Applies a lowpass filter for the imu gyro data.

        :param cutoff: The cutoff frequency for the low-pass filter (in Hz).  
        :param freq: The sampling frequency of the data (in Hz).  
        :param data: A list of gyroscope data to be filtered.   
        :param order: The order of the Butterworth filter (higher values give a steeper filter).
            
        :return: The filtered gyroscope data with high-frequency noise removed.
        """
        assert((isinstance(cutoff, int) or isinstance(cutoff, float)) and cutoff > 0), "The cutoff for filter must be valid number"
        assert((isinstance(freq, int) or isinstance(freq, float)) and freq > 0), "The frequency for filter must be valid number"
        assert(isinstance(data, list) and len(data) > 0), "The input data must be a valid list"
        assert(isinstance(order, int) and order > 0), "Order must be an int greater than 5"
        nyquist = 0.5 * freq
        normal_cuttoff = cutoff / nyquist
        b, a = butter(order, normal_cuttoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def poll(self):
        """
        Calls the read_imu_data() method and polls
        the IMU to read the data that it records.
        Non-threaded polling.
        """
        self.read_imu_data()

    def run(self, sleep_time):
        """
        Calls poll() method to poll, read, and record
        data from the IMU on Artemis OpenLog every 
        sleep_time interval. Non-threaded.

        :param sleep_time: The time interval (in seconds) between each IMU poll()
        """
        assert((isinstance(sleep_time, int) or isinstance(sleep_time, float)) and sleep_time > 0), "Sleep time must be greater than 0" 
        self.poll()
        time.sleep(sleep_time)

    def shutdown(self):
        """
        Terminates communcation and the
        serial port on the Artemis OpenLog.
        """
        self.stop_imu_threading()
        if self.ser:
            self.ser.close()
            logger.info("Closed serial connection")


#---------- Free Functions ----------#

def calibrate_artemis(imu: ArtemisOpenLog, period: int, calibrate_gyro: bool = True, calibrate_accel: bool = True):
    """
    Runs the IMU for a given period time to calculate
    and set the gyro and/or accel offset of the Artemis.

    :param imu: The ArtemisOpenLog IMU to run for a given period of time.
    :param period: The given amount of time to collect samples.
    :param calibrate_gyro: Boolean to determine if gyro calibration is needed.
    :param calibrate_accel: Boolean to determine if accel calibration is needed.

    :return: The gyro and/or accel offset/bias that will be used to calibrate the IMU.
    """
    assert(isinstance(imu, ArtemisOpenLog)), "The imu must be an ArtemisOpenLog object"
    assert(isinstance(period, int) and period > 0), "The period must be a valid int"
    assert(isinstance(calibrate_gyro, bool)), "calibrate_gyro must be a boolean"
    assert(isinstance(calibrate_accel, bool)), "calibrate_accel must be a boolean"
    
    imu.start_imu_threading()

    gyro_x_samples = []
    gyro_y_samples = []
    gyro_z_samples = []
    accel_x_samples = []
    accel_y_samples = []
    accel_z_samples = []

    start_time = time.time()
    while period > time.time() - start_time:
        if calibrate_gyro and imu.gyro_x and imu.gyro_y and imu.gyro_z:
            gyro_x_samples.append(imu.gyro['x'])
            gyro_y_samples.append(imu.gyro['y'])
            gyro_z_samples.append(imu.gyro['z'])
        if calibrate_accel and imu.accel_x and imu.accel_y and imu.accel_z:
            accel_x_samples.append(imu.accel['x'])
            accel_y_samples.append(imu.accel['y'])
            accel_z_samples.append(imu.accel['z'])
        time.sleep(0.01)  # Small delay to avoid excessive CPU usage

    imu.stop_imu_threading()

    if calibrate_gyro:
        gyro_offset = {
            'x': np.mean(gyro_x_samples),
            'y': np.mean(gyro_y_samples),
            'z': np.mean(gyro_z_samples)
        }
        imu.gyro_offset = gyro_offset
    else:
        gyro_offset = None

    if calibrate_accel:
        accel_offset = {
            'x': np.mean(accel_x_samples),
            'y': np.mean(accel_y_samples),
            'z': np.mean(accel_z_samples)
        }
        imu.accel_offset = accel_offset
    else:
        accel_offset = None

    return gyro_offset, accel_offset

def log_data(imu: ArtemisOpenLog, interval_time, log_type: str):
    """
    Logs the recorded data from the given imu object parameter
    into a matplotlib plot in real time for debugging purposes. 

    :param imu: The imu object to plot data fram
    :param interval_time: The time between each data log in the matplotlib
    :param log_type: The type to plot against time. "accel" or "gyro" strings.

    """
    assert(isinstance(imu, ArtemisOpenLog)), "imu parameter must be an ArtemisOpenLog object"
    assert((isinstance(interval_time, int) or isinstance(interval_time, float)) and interval_time > 0), "Interval time must be a valid number"
    assert(isinstance(log_type, str) and (log_type == 'accel' or 'gyro')), "log_type argument must be a string of either 'accel' or 'gyro'"

    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylim(-10, 10) 

    if log_type == "accel":
        ax.set_ylabel("Acceleration (m/s²)")
        line_ax, = ax.plot([], [], label="aX", color="r")
        line_ay, = ax.plot([], [], label="aY", color="g")
        line_az, = ax.plot([], [], label="aZ", color="b")
        ax.legend()

        def update_plot(_):
            line_ax.set_data(range(len(imu.accel_x)), list(imu.accel_x))
            line_ay.set_data(range(len(imu.accel_y)), list(imu.accel_y))
            line_az.set_data(range(len(imu.accel_z)), list(imu.accel_z))

            ax.set_xlim(0, len(imu.accel_x))
            return line_ax, line_ay, line_az
    
    elif log_type == "gyro":
        ax.set_ylabel("Gyroscope (deg/s)")
        line_gx, = ax.plot([], [], label="gX", color="r")
        line_gy, = ax.plot([], [], label="gY", color="g")
        line_gz, = ax.plot([], [], label="gZ", color="b")
        ax.legend()

        def update_plot(_):
            line_gx.set_data(range(len(imu.gyro_x)), list(imu.gyro_x))
            line_gy.set_data(range(len(imu.gyro_y)), list(imu.gyro_y))
            line_gz.set_data(range(len(imu.gyro_z)), list(imu.gyro_z))

            ax.set_xlim(0, len(imu.gyro_x)) 
            return line_gx, line_gy, line_gz

    ani = animation.FuncAnimation(fig, update_plot, interval=interval_time, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":

    SERIAL_PORT = "COM9" # Used Windows OS
    SERIAL_PORT = "/dev/ttyUSB2" # Used for Jetson Nano 
    artemis_imu = ArtemisOpenLog(SERIAL_PORT, 115200, 1)
    
    # Calibrate the IMU first
    #gyro_offset, accel_offset = calibrate_artemis(artemis_imu, 60, calibrate_gyro=True, calibrate_accel=False)
    #logger.info(f"Gyro offset: {gyro_offset}")
    #logger.info(f"Accel offset: {accel_offset}")
    
    artemis_imu.start_imu_threading()
    log_data(artemis_imu, 10, "gyro")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        artemis_imu.shutdown()