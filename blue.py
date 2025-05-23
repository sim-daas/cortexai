import serial
import time

# --- Configuration ---
SERIAL_PORT = "/dev/rfcomm0"  # Or "COMx" on Windows
BAUD_RATE = 9600              # MUST match Arduino's BTSerial.begin() baud rate
# ---------------------

arduino_bt = None  # Initialize to None

try:
    print(f"Attempting to connect to {SERIAL_PORT} at {BAUD_RATE} baud...")
    # Increased timeout for readline on the Arduino side if you plan to send back replies
    arduino_bt = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Successfully connected to Arduino via Bluetooth.")
    print("Enter messages to send to Arduino. Type 'exit' or 'quit' to close.")
    time.sleep(0.5) # Give a moment for the connection to stabilize and Arduino to be ready

    while True:
        message_to_send = input("Laptop_TX> ")
        if message_to_send.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        # Add a newline character so Arduino's .readStringUntil('\n') or
        # multiple .read() calls followed by Serial.println() work well.
        message_with_newline = message_to_send + '\n'

        try:
            arduino_bt.write(message_with_newline.encode('utf-8'))
            # print(f"Sent: '{message_to_send}'") # Optional: confirmation of sending
        except serial.SerialException as write_error:
            print(f"Error writing to serial port: {write_error}")
            print("Connection may have been lost. Exiting.")
            break
        except Exception as e_write:
            print(f"An unexpected error occurred during write: {e_write}")
            break

        # Optional: Add a small delay if sending very rapidly,
        # though usually not necessary for typed input.
        # time.sleep(0.05)

        # If you also want to listen for immediate replies from Arduino here,
        # you would add a non-blocking read or a short blocking read.
        # For simplicity, this example focuses on sending.
        # Example for trying to read a quick reply:
        # if arduino_bt.in_waiting > 0:
        #     try:
        #         reply = arduino_bt.readline().decode('utf-8').strip()
        #         if reply:
        #             print(f"Arduino_RX< {reply}")
        #     except Exception as e_read:
        #         print(f"Error reading reply: {e_read}")


except serial.SerialException as e:
    print(f"Serial Connection Error: {e}")
    print("Troubleshooting:")
    print(f"- Is the device at {SERIAL_PORT} available and paired?")
    print("- Is the Arduino/Bluetooth module powered on?")
    print(f"- Is another program using {SERIAL_PORT}?")
except KeyboardInterrupt:
    print("\nExiting due to user interrupt (Ctrl+C)...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if arduino_bt and arduino_bt.is_open:
        print("Closing Bluetooth connection.")
        arduino_bt.close()
