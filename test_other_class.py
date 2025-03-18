import logging as logger
import serial
import time

class ArtemisOpenLog:
    def __init__(self, port, baudrate, timeout):
        assert(isinstance(port, str)), "Port must be a valid string input"
        assert(isinstance(baudrate, int) and baudrate > 0), "Baudrate must be valid int"
        assert((isinstance(timeout, int) or isinstance(timeout, float)) and timeout > 0), "timeout time must be valid number greater than 0" 
        self.ser = None
        self.accel = { 'x': 0, 'y': 0, 'z': 0}
        self.gyro = { 'x' : 0, 'y': 0, 'z': 0}
        self.euler = { 'x': 0, 'y': 0, 'z': 0}
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)  # Fix: pass timeout as a keyword argument
            logger.info(f"Connected to {port}")
        except serial.SerialException as e:
            logger.error(f"Error: {e}")
        except KeyboardInterrupt:
            logger.info("\nExiting...")

    def poll(self):
        data = self.ser.readline().decode('utf-8').strip().split(",")
        try:
            self.accel = {
                'x' : float(data[2]) * 0.00981,
                'y' : float(data[3]) * 0.00981,
                'z' : float(data[4]) * 0.00981
            }
            self.gyro = {
                'x': float(data[5]),
                'y': float(data[6]),
                'z': float(data[7])
            }
        except IndexError:
            logger.error("Waiting for IMU data to parse")
        except ValueError:
            logger.error("Waiting for IMU data to cast to floats")

    def run(self):
        self.poll()
        return self.gyro, self.accel

    def run_threaded(self):
        return self.gyro, self.accel

    def shutdown(self):
        if self.ser:
            self.ser.close()
            logger.info("Closed serial connection")

if __name__ == "__main__":

    SERIAL_PORT = "COM9" # Used Windows OS
    #SERIAL_PORT = "/dev/ttyUSB2" # Used for Jetson Nano 
    artemis_imu = ArtemisOpenLog(SERIAL_PORT, 115200, 1)

    try:
        while True:
            accel, gyro = artemis_imu.run()
            print(f"Gyro: {gyro}, Accel: {accel}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        artemis_imu.shutdown()

