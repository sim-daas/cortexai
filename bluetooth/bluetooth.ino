#include <SoftwareSerial.h>

// Define the pins for SoftwareSerial
// Arduino Pin 10 is RX for SoftwareSerial (connects to BT Module's TX)
// Arduino Pin 11 is TX for SoftwareSerial (connects to BT Module's RX via voltage divider if needed)
const byte bluetoothRxPin = 10; // This is Arduino's RX pin for BT communication
const byte bluetoothTxPin = 11; // This is Arduino's TX pin for BT communication

SoftwareSerial BTSerial(bluetoothRxPin, bluetoothTxPin); // Arduino RX, Arduino TX

unsigned long previousMillis = 0;
const long interval = 2000; // Interval at which to send message (milliseconds)
int counter = 0;

void setup() {
  // Initialize serial communication for the Serial Monitor (USB)
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("Arduino Serial Monitor Initialized.");
  Serial.println("Attempting to send data to Bluetooth device...");
  Serial.println("Also listening for incoming Bluetooth data.");

  // Initialize serial communication for the Bluetooth module
  // !!! IMPORTANT: Make sure this baud rate matches your Bluetooth module's configuration !!!
  // Common baud rates for HC-05/HC-06 are 9600 or 38400.
  BTSerial.begin(9600);
}

void loop() {
  // Send a message to Bluetooth periodically
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    String messageToSend = "Hello from Arduino! Count: " + String(counter);
    BTSerial.println(messageToSend); // Send with a newline

    // Also print to Serial Monitor what we're sending, for debugging
    Serial.print("Sent via BT: ");
    Serial.println(messageToSend);

    counter++;
  }

  // Check if data is available from the Bluetooth module (sent from laptop)
  if (BTSerial.available()) {
    Serial.print("Received via BT: ");
    while (BTSerial.available()) { // Read all available characters
      char receivedChar = (char)BTSerial.read();
      Serial.print(receivedChar); // Print to Serial Monitor
    }
    Serial.println(); // Add a newline in the Serial Monitor after printing received data
  }
}