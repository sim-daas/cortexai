#include <SoftwareSerial.h>
#include <Servo.h> // Include the Servo library

// Define the pins for SoftwareSerial
// Arduino Pin 10 is RX for SoftwareSerial (connects to BT Module's TX)
// Arduino Pin 11 is TX for SoftwareSerial (connects to BT Module's RX via voltage divider if needed)
const byte bluetoothRxPin = 10; // This is Arduino's RX pin for BT communication
const byte bluetoothTxPin = 11; // This is Arduino's TX pin for BT communication

const int servoPin = 6; // Define the servo pin
Servo myServo;          // Create a Servo object

SoftwareSerial BTSerial(bluetoothRxPin, bluetoothTxPin); // Arduino RX, Arduino TX

unsigned long previousMillis = 0;
const long interval = 2000; // Interval at which to send message (milliseconds)
int counter = 0;
String btReceivedString = ""; // String to store received Bluetooth data

void setup()
{
  // Initialize serial communication for the Serial Monitor (USB)
  Serial.begin(9600);
  while (!Serial)
  {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("Arduino Serial Monitor Initialized.");
  Serial.println("Attempting to send data to Bluetooth device...");
  Serial.println("Also listening for incoming Bluetooth data (angle 0-170).");

  myServo.attach(servoPin); // Attach the servo on pin 6
  myServo.write(90);        // Initialize servo to a neutral position (optional)
  Serial.println("Servo initialized and attached to pin " + String(servoPin));

  // Initialize serial communication for the Bluetooth module
  // !!! IMPORTANT: Make sure this baud rate matches your Bluetooth module's configuration !!!
  // Common baud rates for HC-05/HC-06 are 9600 or 38400.
  BTSerial.begin(9600);
}

void loop()
{
  // Send a message to Bluetooth periodically
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;

    String messageToSend = "Hello from Arduino! Count: " + String(counter);
    BTSerial.println(messageToSend); // Send with a newline

    // Also print to Serial Monitor what we're sending, for debugging
    Serial.print("Sent via BT: ");
    Serial.println(messageToSend);

    counter++;
  }

  // Check if data is available from the Bluetooth module (sent from laptop)
  if (BTSerial.available())
  {
    char receivedChar = (char)BTSerial.read();
    btReceivedString += receivedChar; // Append the character to our string

    // If a newline character is received, the command is complete
    if (receivedChar == '\n')
    {
      btReceivedString.trim(); // Remove any leading/trailing whitespace (including \r)
      Serial.print("Received via BT: ");
      Serial.println(btReceivedString);

      // Attempt to convert the string to an integer (angle)
      int angle = btReceivedString.toInt();

      // Check if the conversion was successful and the angle is within range
      if (btReceivedString.length() > 0 && angle >= 0 && angle <= 170)
      {
        Serial.print("Setting servo to angle: ");
        Serial.println(angle);
        myServo.write(angle); // Move the servo to the specified angle
      }
      else if (btReceivedString.length() > 0)
      {
        Serial.print("Invalid angle received or out of range (0-170): ");
        Serial.println(btReceivedString);
      }

      btReceivedString = ""; // Clear the string for the next command
    }
  }
}