import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class EKF_IMU:
    def __init__(self, dt):
        self.dt = dt  
        self.x = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion
        self.P = np.eye(4) * 0.01  
        self.Q = np.eye(4) * 1e-5  
        self.R_acc = np.eye(3) * 0.01  
        self.R_mag = np.eye(3) * 10  

    def predict(self, gyro):
        wx, wy, wz = gyro  
        omega = np.array([
            [ 0, -wx, -wy, -wz],
            [ wx,  0,  wz, -wy],
            [ wy, -wz,  0,  wx],
            [ wz,  wy, -wx,  0]
        ])
        dq = 0.5 * omega @ self.x * self.dt
        self.x += dq
        self.x /= np.linalg.norm(self.x)
        self.P += self.Q  

    def update(self, acc, mag):
        acc = acc / np.linalg.norm(acc)  # Normalize accelerometer data
        mag = mag / np.linalg.norm(mag)  # Normalize magnetometer data

        g_ref = np.array([0, 0, 1])
        g_meas = self.quat_rotate(self.x, g_ref)
        acc_error = acc - g_meas

        m_ref = np.array([1, 0, 0])
        m_meas = self.quat_rotate(self.x, m_ref)
        mag_error = mag - m_meas

        y = np.hstack((acc_error, mag_error))
        H = np.zeros((6, 4)) 
        H[:3, :3] = np.eye(3)
        H[3:, :3] = np.eye(3)

        R = np.block([[self.R_acc, np.zeros((3,3))], [np.zeros((3,3)), self.R_mag]])  
        S = H @ self.P @ H.T + R  
        K = self.P @ H.T @ np.linalg.inv(S)  

        self.x += K @ y
        self.x /= np.linalg.norm(self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def quat_rotate(self, q, v):
        r = R.from_quat(q)
        return r.apply(v)

    def get_euler_angles(self):
        r = R.from_quat(self.x)
        return r.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw

# --- Dynamic Sensor Data Processing ---
if __name__ == "__main__":
    ekf = EKF_IMU(dt=0.01)

    # Simulating sensor readings over time
    sensor_data = [
        (np.array([0.12, -0.25, 2.45]), np.array([0.02, -0.01, 9.79]), np.array([0.48, 0.12, 0.29])),
        (np.array([-0.14, 0.08, 2.65]), np.array([0.05, 0.00, 9.74]), np.array([0.52, 0.10, 0.31])),
        (np.array([0.20, -0.12, 2.80]), np.array([-0.02, 0.03, 9.81]), np.array([0.49, 0.11, 0.30])),
        (np.array([0.00, 0.00, 3.05]), np.array([0.07, -0.04, 9.76]), np.array([0.51, 0.13, 0.28])),
        (np.array([-0.10, 0.15, 2.90]), np.array([0.00, 0.00, 9.82]), np.array([0.50, 0.09, 0.32])),
        (np.array([0.18, -0.20, 2.55]), np.array([0.06, -0.02, 9.78]), np.array([0.48, 0.14, 0.27])),
        (np.array([0.05, 0.05, 3.10]), np.array([-0.03, 0.02, 9.80]), np.array([0.53, 0.08, 0.33])),
        (np.array([-0.12, 0.22, 2.75]), np.array([0.02, -0.03, 9.79]), np.array([0.50, 0.10, 0.31])),
        (np.array([0.25, -0.18, 2.65]), np.array([0.04, 0.00, 9.77]), np.array([0.49, 0.12, 0.29])),
        (np.array([-0.08, 0.10, 2.95]), np.array([0.00, -0.02, 9.81]), np.array([0.52, 0.11, 0.30])),
    ]

    for i, (gyro, acc, mag) in enumerate(sensor_data):
        print(f"\nIteration {i+1}")
        ekf.predict(gyro)      # Gyroscope update
        ekf.update(acc, mag)   # Accelerometer & Magnetometer correction
        euler_angles = ekf.get_euler_angles()
        
        print(f"Gyro: {gyro}, Acc: {acc}, Mag: {mag}")
        print(f"Roll: {euler_angles[0]:.2f}°, Pitch: {euler_angles[1]:.2f}°, Yaw: {euler_angles[2]:.2f}°")
        
        time.sleep(0.5)  # Simulate real-time updates
