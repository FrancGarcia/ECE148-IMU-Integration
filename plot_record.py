import serial
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

SERIAL_PORT = "COM9"  # Change this to match your system
BAUD_RATE = 115200
NUM_DATA_POINTS = 14
BUFFER_SIZE = 100 

ax_data = deque(maxlen=BUFFER_SIZE)
ay_data = deque(maxlen=BUFFER_SIZE)
az_data = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")

    with open('imu_data.csv', 'w', newline='') as fp_imu:
        writer = csv.writer(fp_imu)
        writer.writerow(["rtcDate", "rtcTime", "aX", "aY", "aZ", "gX", "gY", "gZ", "mX", "mY", "mZ", "imu_degC", "output_Hz"])  # Header

        fig, ax = plt.subplots()
        ax.set_title("Real-Time IMU Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Acceleration (mm/s²)")
        ax.set_ylim(-10, 10)  

        line_ax, = ax.plot([], [], label="aX", color="r")
        line_ay, = ax.plot([], [], label="aY", color="g")
        line_az, = ax.plot([], [], label="aZ", color="b")
        ax.legend()

        def update_plot(frame):
            """Updates the plot with new data"""
            data = ser.readline().decode('utf-8').strip().split(",")

            if len(data) == NUM_DATA_POINTS:
                try:
                    timestamps.append(data[1])
                    ax_data.append(float(data[2]) * 0.00981)  
                    ay_data.append(float(data[3]) * 0.00981)  
                    az_data.append(float(data[4]) * 0.00981) 

                    # Acceleration units: mg = m/s^2 * 0.00981
                    writer.writerow(data[:-1]) 
                    print("".join(data[:-1]))

                    line_ax.set_data(range(len(ax_data)), ax_data)
                    line_ay.set_data(range(len(ay_data)), ay_data)
                    line_az.set_data(range(len(az_data)), az_data)

                    ax.set_xlim(0, len(ax_data)) 
                    return line_ax, line_ay, line_az

                except ValueError:
                    print(f"Skipping invalid data: {data}")

        # Set up real-time plotting --> smaller interval = more responsive plotting
        ani = animation.FuncAnimation(fig, update_plot, interval=5, cache_frame_data=False)

        plt.show()

except serial.serialutil.SerialException as e:
    print(f"Serial Error: {e}")

except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
