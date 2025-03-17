import serial
import csv

def record_imu_data():
    """
    Records IMU data from COM port and logs it onto a CSV file.
    """
    # Replace 'COMX' (Windows) or '/dev/ttyUSB0' (Linux/macOS) with your actual port
    SERIAL_PORT = "COM9"  # COM port may change depending on device
    BAUD_RATE = 115200 
    NUM_DATA_POINTS = 14

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")

        with open('imu_data.csv', 'w', newline='') as fp_imu:
            writer = csv.writer(fp_imu)
            while True:
                data_array = ser.readline().decode('utf-8').strip().split(",")
                if len(data_array) == NUM_DATA_POINTS:
                    # Data format:
                    # rtcDate,rtcTime,aX,aY,aZ,gX,gY,gZ,mX,mY,mZ,imu_degC,output_Hz,""
                    # Drop the last column and convert to a string to write to csv file
                    data_array = data_array[0:len(data_array)-1]
                    print("".join(data_array))
                    writer.writerow(data_array)
                    print(",".join(data_array))

    except serial.SerialException as e:
        print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")
        ser.close()
    
if __name__ == "__main__":
    record_imu_data()
