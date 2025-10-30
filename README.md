# **PlutoSDR PSK/QAM Modulation Analyzer**

This project provides a comprehensive graphical user interface (GUI) for demonstrating and analyzing the real-world performance of **PSK (Phase-Shift Keying)** and **QAM (Quadrature Amplitude Modulation)** using an **Analog Devices PlutoSDR**.

It allows you to transmit a modulated signal over the air, receive it, perform robust frame synchronization and channel estimation, and then plot detailed performance metrics like constellation diagrams and Symbol Error Rate (SER) curves.

## **Features**

* **Dual Modulation Support:** Run identical experiments for both M-PSK and M-QAM.  
* **Interactive GUI:** A tkinter-based interface allows you to select which tests to run and configure all parameters without editing code.  
* **Real-Time SDR TX/RX:** Transmits and receives real signals using a PlutoSDR.  
* **Robust Receiver Chain:** Implements a full receiver pipeline:  
  * **Frame Synchronization:** Uses a Barker-code-based sequence to find the start of the frame in the received buffer.  
  * **Channel Estimation:** Uses a known pilot sequence to estimate the complex gain and phase shift (h\_est) of the wireless channel.  
  * **Equalization:** Corrects the received data symbols by dividing out the estimated channel effect.  
* **Detailed Performance Analysis:** Run multiple experiments from the GUI:  
  * **Single Frame Demo:** See detailed plots (constellations, correlation peaks) for one TX/RX cycle.  
  * **SER vs. SPS:** Sweep the number of samples-per-symbol.  
  * **SER vs. Modulation Order (M):** Compare BPSK, QPSK, 8-PSK, 16-QAM, etc.  
  * **SER vs. Frame Length (N):** Analyze the effect of frame size.  
  * **SER vs. Sync Length:** Test the impact of different pilot lengths.  
* **Safe Threading:** All SDR operations run in a background thread to keep the GUI responsive and display live log output.

## **File Structure**

The project is divided into four main files:

| File | Role |
| :---- | :---- |
| **maincode.py** | **(Main Entry Point)** A simple launcher window that lets you choose to run either the PSK or QAM demo. |
| **pluto\_only\_PSK.py** | A complete, self-contained application for all PSK-related experiments. Includes the PSK modulator/demodulator and the main experiment GUI. |
| **pluto\_only\_QAM.py** | A complete, self-contained application for all QAM-related experiments. Includes the QAM modulator/demodulator and the main experiment GUI. |
| software\_only\_PSK.py | (Not run by launcher) The original pure-software simulation used to develop the signal processing chain. It simulates the channel and noise instead of using an SDR. |

## **Installation and Usage**

### **Installation**

**1\. Hardware:**

* You must have an **Analog Devices PlutoSDR** connected to your computer via USB.

**2\. Software:**

* You will need Python 3 and the following libraries:  
  * **numpy**: For all numerical and signal processing.  
  * **matplotlib**: For generating plots of the results.  
  * **pyadi-iio**: The official Analog Devices library to communicate with the PlutoSDR.  
  * **tkinter**: For the GUI (usually included with standard Python).  
* You can install the required Python libraries using pip:  
  pip install numpy matplotlib pyadi-iio

### **How to Run**

1. **Connect your PlutoSDR** and ensure it is reachable on your network. The default IP is ip:192.168.2.1.  
   * *Note:* If your Pluto has a different IP, you must update the PLUTO\_IP variable at the top of pluto\_only\_PSK.py and pluto\_only\_QAM.py.  
2. **Run the main launcher** from your terminal:  
   python maincode.py

3. **Select a modulation:** The "SDR Demo Launcher" window will appear. Choose either "PSK" or "QAM" and click **"Run Selected Demo"**.  
4. **Configure and run experiments:** The main experiment window will open.  
   * **Hardware Setup:** Set your global tx\_gain, rx\_gain\_mode, rx\_gain, and Pluto center freq (MHz).  
   * **Select Tests:** Check the box for any experiment you want to run (e.g., "Run single-frame demo"). You can select multiple tests.  
   * **Adjust Parameters:** Configure the settings for each test you selected.  
   * **Execute:** Click the **"Run selected simulations"** button at the bottom.  
5. **View Results:**  
   * A new window titled "Run Output" will appear, showing the live log output from the experiment.  
   * Once the experiments are finished, matplotlib plot windows will open displaying the results (constellation diagrams, SER curves, etc.).

## **Core Logic: Frame Structure & Processing**

This project's core logic relies on a custom-built frame structure to enable robust synchronization and channel estimation.

### **1\. Frame Generation (TX)**

Each transmitted frame is built by concatenating three components:

* **\[ SYNC \]**: A known synchronization sequence (based on a Barker code) used by the receiver to find the start of the frame.  
* **\[ PILOT \]**: A sequence of known, pre-defined modulation symbols (e.g., 128 random 16-PSK symbols).  
* **\[ DATA \]**: The payload, consisting of random data symbols.

This \[SYNC | PILOT | DATA\] frame is then upsampled by sps (samples-per-symbol) and transmitted by the PlutoSDR.

### **2\. Receiver Processing (RX)**

When the PlutoSDR receives the signal, it performs the following steps:

1. **Synchronization:** The receiver correlates the incoming signal with the known \[ SYNC \] sequence. The location of the highest correlation peak tells the receiver exactly where the frame begins.  
2. **Channel Estimation:** The receiver extracts the \[ PILOT \] symbols from the received signal. It then compares these *received* symbols to the *known, original* pilot symbols. By averaging the difference (e.g., h\_est \= mean(rx\_pilots / tx\_pilots)), it calculates the average complex gain (h\_est) representing the channel's distortion.  
3. **Equalization:** The receiver extracts the \[ DATA \] symbols. To correct for the channel, it simply divides each received data symbol by the channel estimate: corrected\_symbol \= received\_symbol / h\_est.  
4. **Demodulation:** The corrected\_symbol is passed to the PSK or QAM demodulator, which decides which symbol was most likely sent. The results are compared to the original transmitted data to calculate the Symbol Error Rate (SER).