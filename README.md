# TEACHER CORRECTION ASSISTANCE GROUP 4

BERDNYK Mariia - IA/ID

HADDOU Amine - IA/ID

SOP DJONKAM Karl Alwyn  - IA/ID

---

BENNA Bachir - IoT

ESSAM Steven - IoT

---

To process images and extract text using either a Raspberry Pi or a computer with CPU/GPU support.

---

## Running on Raspberry Pi

> **Note**: The Raspberry Pi's CPU may not be powerful enough for this process. The operating system may terminate the process to protect runtime stability.

### Steps:

1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   ```

2. **Install Dependencies**  
   Install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Network**  
   - Set up an access point on your computer.  
   - While building the Raspberry Pi OS image, configure the SSID and password for your network.

4. **Run the Scripts**  
   - Quantize the model:  
     ```bash
     python quantisize.py
     ```  
   - Start the server:  
     ```bash
     python lib/server.py
     ```  
   - Run the IoT interface:  
     ```bash
     python lib/static/images/iotInterface.py
     ```  
     **Note**: Before running `iotInterface.py`, ensure the following in `lib/static/script.js`:  
     - The variable `this_ip` should contain the IP address of your Raspberry Pi.  
     - Do not change the port.

5. **Access the Interface**  
   On your computer, navigate to:  
   ```
   http://<raspberrypi_ip>:7860
   ```  
   - Use the interface to take a picture.  
   - Click "Next" to send the image for processing.

6. **Important Notes**  
   - When clicking "Extract Text," the Raspberry Pi may become unresponsive until the process is terminated by the OS to protect runtime stability.

---

## Running on Your Computer (CPU or GPU)

### Steps:

1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Scripts**  
   - Quantize the model:  
     ```bash
     python quantisize.py
     ```  
   - Start the server:  
     ```bash
     python lib/server.py
     ```
   - Go to your_localhost:5000  

4. **Choose CPU or GPU**  
   - The models will run based on your choice (CPU or GPU) using the interface.

---
