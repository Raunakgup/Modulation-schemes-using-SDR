import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sys
import queue

# Pluto-only robust channel estimation demo (cleaned)
# --------------------------------------------------
# This version is *Pluto SDR only*. All simulation helpers and any software
# AWGN injection have been removed; the code always transmits the generated
# frames on an attached ADI PlutoSDR and receives the over-the-air capture.
#
# Important: make sure PLUTO_IP and PLUTO_CENTER_FREQ match your hardware.

# --------------------------- User-configurable globals ----------------------
PLUTO_IP = "ip:192.168.2.1"  # Pluto device URI (change as needed)
PLUTO_CENTER_FREQ = 915e6    # RF center frequency in Hz

# GUI-controlled defaults (updated by popup)
GUI_TX_GAIN = -40.0  # dB
GUI_RX_GAIN = 0.0    # dB
GUI_RX_GAIN_MODE = 'manual' # 'manual', 'slow_attack', or 'fast_attack'
GUI_CENTER_FREQ = PLUTO_CENTER_FREQ  # Hz (popup uses MHz)

# We import the Pluto driver (pyadi-iio) only when running the script
import time
import adi

# Optional fixed RNG seed for reproducibility during debugging; remove to
# allow different frames each run.
np.random.seed(1)

# --------------------------- Utility helper functions ----------------------

def print_snip(name, x):
    """Print a short slice of an array for quick debugging.

    Prints up to the first 12 elements in a compact complex format.
    """
    if x is None:
        return
    x = np.asarray(x)
    N = min(12, x.size)
    xs = x.ravel()[:N]

    def fmt(v):
        re = np.real(v)
        im = np.imag(v)
        return f"{re:.4g}{im:+.4g}j"
    strs = ", ".join(fmt(v) for v in xs)
    print(f"INFO: [{name}] len={x.size}, first={{ {strs} }}")


def pskmod(symbols, M=2):
    """M-PSK modulator: map integer symbols -> complex constellation.

    Returns complex128 array of constellation points.
    """
    syms = np.asarray(symbols).astype(int).ravel()
    syms = np.mod(syms, M)
    return np.exp(1j * 2.0 * np.pi * syms / M).astype(np.complex128)


def pskdemod(symbols, M=2):
    """M-PSK demodulator via minimum Euclidean distance to constellation.

    Returns integer decisions in 0..M-1.
    """
    symbols = np.asarray(symbols).ravel().astype(np.complex128)
    const = np.exp(1j * 2.0 * np.pi * np.arange(M) / M).astype(np.complex128)
    diff = symbols.reshape(-1, 1) - const.reshape(1, -1)
    dist = np.abs(diff)
    decisions = np.argmin(dist, axis=1).astype(int)
    return decisions


def biterr(a, b):
    """Count symbol errors and return (num_errors, ser).

    a and b should be integer arrays of identical length.
    """
    a = np.asarray(a).ravel().astype(int)
    b = np.asarray(b).ravel().astype(int)
    errs = np.sum(a != b)
    ser = errs / a.size
    return int(errs), float(ser)


# --------------------------- PlutoSDR wrapper -------------------------------
# Small wrapper around adi.Pluto that exposes only the operations used by
# this script: set sample rate/LO, set gains, transmit cyclic waveform, read
# a frame-sized RX buffer, flush RX buffers, and cleanup.

class PlutoSDRWrapper:
    """Minimal Pluto helper for single-frame TX/RX experiments.

    Behavior notes:
      - Constructor connects to the device and sets sample-rate and LO.
      - tx_waveform expects a complex numpy array scaled to Pluto's expected
        amplitude convention (we follow the earlier code and scale by 2**14).
      - tx_waveform(cyclic=True) enables continuous looped transmission
        until stop_tx() is called.
      - rx_once() returns rx_buffer_size complex samples (as numpy array).
    """

    def __init__(self, ip=PLUTO_IP, sample_rate=1e6, center_freq=PLUTO_CENTER_FREQ, rx_buffer_size=1024):
        # Connect to Pluto (raises if device not reachable)
        self.sdr = adi.Pluto(ip)
        self.sample_rate = int(sample_rate)
        self.center_freq = int(center_freq)
        self.sdr.sample_rate = self.sample_rate

        # Default RF settings (caller may alter after constructing)
        self.sdr.tx_rf_bandwidth = self.sample_rate
        self.sdr.tx_lo = self.center_freq
        self.sdr.rx_lo = self.center_freq
        self.sdr.rx_rf_bandwidth = self.sample_rate

        # Tell the driver how many samples to return per rx() call
        self.sdr.rx_buffer_size = int(rx_buffer_size)

    def set_tx_gain(self, db):
        # TX hardware gain (chan0)
        self.sdr.tx_hardwaregain_chan0 = float(db)

    def set_rx_gain(self, db, mode='manual'):
        # RX gain control: prefer 'manual' for stable repeatable estimates
        self.sdr.gain_control_mode_chan0 = mode
        if mode == 'manual':
            self.sdr.rx_hardwaregain_chan0 = float(db)

    def tx_waveform(self, samples, cyclic=True):
        # Start transmitting the provided complex numpy array
        self.sdr.tx_cyclic_buffer = bool(cyclic)
        self.sdr.tx(samples)

    def stop_tx(self):
        # Stop and destroy TX buffer if possible
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

    def rx_once(self):
        # Receive a single buffer (length = rx_buffer_size)
        return self.sdr.rx()

    def rx_flush(self, n=5, delay_s=0.01):
        # Flush 'n' buffers with small delay to allow refilling
        for _ in range(int(n)):
            _ = self.sdr.rx()
            time.sleep(delay_s)

    def close(self):
        try:
            self.stop_tx()
        except Exception:
            pass
        del self.sdr


# --------------------------- Frame construction ----------------------------
DEFAULT_BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=int)

def _make_sync_bits(sync_len_bits, barker=DEFAULT_BARKER13):
    """Create a binary sync sequence (0/1) of requested length by repeating
    the base Barker sequence.
    """
    reps = ceil(sync_len_bits / len(barker))
    seq = np.tile(barker, reps)[:sync_len_bits]
    return (seq < 0).astype(int)


def run_single_frame_demo(Nsymbols=4000, M=16, sps=8, fs=1e6,
                          sync_barker13=None, ch_pilot_len_bits=128,
                          seed=None, pluto_ip=PLUTO_IP):
    """Construct and transmit a single frame using Pluto, receive the frame
    capture and perform synchronization, channel estimation and demodulation.

    Returns a dictionary of results and arrays for plotting.
    """
    if seed is not None:
        np.random.seed(seed)

    if sync_barker13 is None:
        sync_barker13 = DEFAULT_BARKER13
    sync_barker13 = np.tile(sync_barker13, 3)
    sync_len_bits = len(sync_barker13)

    print(f"INFO: [PARAM] Nsymbols={Nsymbols}, M={M}, sps={sps}, sync_len={sync_len_bits}, chpilot_len={ch_pilot_len_bits}")

    # Build frame pieces (symbol indices)
    sync_bits = (sync_barker13 < 0).astype(int)
    chpilot_bits = np.random.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = np.random.randint(0, M, size=(Nsymbols,))

    # Map to constellation symbols
    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    # NRZ shaping: repeat each symbol sps times
    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    # Concatenate into a single transmit buffer
    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nsymbols
    frame_samples = frame_bits * sps
    print(f"INFO: [TX] frame_bits={frame_bits}, frame_samples={frame_samples}")
    print_snip('tx_frame (first)', tx_frame[:min(12, tx_frame.size)])

    # ------------------- Transmit + Receive on Pluto ----------------------
    print("INFO: [PLUTO] Initializing PlutoSDR and transmitting frame...")
    pluto = PlutoSDRWrapper(ip=pluto_ip, sample_rate=fs, center_freq=GUI_CENTER_FREQ, rx_buffer_size=frame_samples)

    # Set gains from GUI
    pluto.set_tx_gain(GUI_TX_GAIN)
    pluto.set_rx_gain(GUI_RX_GAIN, mode=GUI_RX_GAIN_MODE)

    # Scale to Pluto amplitude convention and start cyclic TX
    tx_samples = (tx_frame * (2**14)).astype(np.complex64)
    pluto.tx_waveform(tx_samples, cyclic=True)

    # Flush and capture one frame-sized buffer
    pluto.rx_flush(n=5)
    rx_samples = pluto.rx_once()

    # Stop TX and cleanup
    pluto.stop_tx()
    pluto.close()

    # Use received samples as the real-channel data (do not add software noise)
    rx_no_noise = rx_samples.astype(np.complex128)
    rx_awgn = rx_no_noise  # retained name for compatibility with downstream variable usage
    print_snip('rx_no_noise (first)', rx_no_noise[:min(12, rx_no_noise.size)])

    # ------------------------- Downstream processing ----------------------
    # Correlate for sync using a doubled buffer to handle wrap-around
    rx2_no = np.concatenate([rx_no_noise, rx_no_noise])
    corr_no = np.abs(np.convolve(rx2_no, np.conjugate(sync_shaped[::-1])))
    peak_no = np.argmax(corr_no)
    pilot_start_in_rx2_no = peak_no - (len(sync_shaped) - 1)
    pilot_start_no = pilot_start_in_rx2_no % frame_samples
    print(f"INFO: [SYNC_NO] peak_no={peak_no}, pilot_start_no={pilot_start_no+1}, pilot_start mod sps={(pilot_start_no % sps)+1}")

    seglen_no = min(len(sync_shaped), frame_samples - pilot_start_no)
    seg_no = rx_no_noise[pilot_start_no: (pilot_start_no + seglen_no)]
    match_frac_no = np.sum(np.abs(seg_no - sync_shaped[:seglen_no]) < 1e-6) / seglen_no
    print(f"INFO: [SYNC_MATCH_NO] seglen={seglen_no}, match_frac={match_frac_no:.4f}")

    # Channel pilot extraction and channel estimation
    chpilot_start_no = pilot_start_no + len(sync_shaped)
    chpilot_indices_no = (np.arange(chpilot_start_no, chpilot_start_no + ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_no = rx_no_noise[chpilot_indices_no]
    chpilot_mat_no = chpilot_samples_no.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_no = np.sum(chpilot_mat_no, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_no (first)', rx_chpilot_symbols_no[:min(12, rx_chpilot_symbols_no.size)])

    h_est_no = np.mean(rx_chpilot_symbols_no / chpilot_symbols)
    print(f"INFO: [H_EST_NO] h_est_no={h_est_no.real:.4f}{h_est_no.imag:+.4f}j")

    # Data extraction, equalization and demodulation (no software AWGN)
    data_start_no = chpilot_start_no + ch_pilot_len_bits * sps
    data_indices_no = (np.arange(data_start_no, data_start_no + Nsymbols*sps) % frame_samples).astype(int)
    data_samples_no = rx_no_noise[data_indices_no]
    data_mat_no = data_samples_no.reshape((sps, Nsymbols), order='F')
    symbol_samples_no = np.sum(data_mat_no, axis=0).astype(np.complex128)
    print_snip('symbol_samples_no (first)', symbol_samples_no[:min(12, symbol_samples_no.size)])

    rx_syms_before_no = pskdemod(symbol_samples_no, M)
    errs_before_no, ser_before_no = biterr(data_bits, rx_syms_before_no)
    print(f"INFO: [DEM_BEFORE_NO] NumErr={errs_before_no}, SER={ser_before_no:.6g}")

    symbol_samples_eq_no = symbol_samples_no / h_est_no
    print_snip('symbol_samples_eq_no (first)', symbol_samples_eq_no[:min(12, symbol_samples_eq_no.size)])

    rx_syms_after_no = pskdemod(symbol_samples_eq_no, M)
    errs_after_no, ser_after_no = biterr(data_bits, rx_syms_after_no)
    print(f"INFO: [DEM_AFTER_NO] NumErr={errs_after_no}, SER={ser_after_no:.6g}")

    # Repeat the same pipeline names for 'received buffer' variant to preserve
    # plotting code layout from the original script.
    rx2_aw = np.concatenate([rx_awgn, rx_awgn])
    corr_aw = np.abs(np.convolve(rx2_aw, np.conjugate(sync_shaped[::-1])))
    peak_aw = np.argmax(corr_aw)
    pilot_start_in_rx2_aw = peak_aw - (len(sync_shaped) - 1)
    pilot_start_aw = pilot_start_in_rx2_aw % frame_samples
    print(f"INFO: [SYNC_AW] peak_aw={peak_aw}, pilot_start_aw={pilot_start_aw+1}, pilot_start mod sps={(pilot_start_aw % sps)+1}")

    seglen_aw = min(len(sync_shaped), frame_samples - pilot_start_aw)
    seg_aw = rx_awgn[pilot_start_aw: (pilot_start_aw + seglen_aw)]
    match_frac_aw = np.sum(np.abs(seg_aw - sync_shaped[:seglen_aw]) < 1e-6) / seglen_aw
    print(f"INFO: [SYNC_MATCH_AW] seglen={seglen_aw}, match_frac={match_frac_aw:.4f}")

    chpilot_start_aw = pilot_start_aw + len(sync_shaped)
    chpilot_indices_aw = (np.arange(chpilot_start_aw, chpilot_start_aw + ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_aw = rx_awgn[chpilot_indices_aw]
    chpilot_mat_aw = chpilot_samples_aw.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_aw = np.sum(chpilot_mat_aw, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_aw (first)', rx_chpilot_symbols_aw[:min(12, rx_chpilot_symbols_aw.size)])

    h_est_aw = np.mean(rx_chpilot_symbols_aw / chpilot_symbols)
    print(f"INFO: [H_EST_AW] h_est_aw={h_est_aw.real:.4f}{h_est_aw.imag:+.4f}j")

    data_start_aw = chpilot_start_aw + ch_pilot_len_bits * sps
    data_indices_aw = (np.arange(data_start_aw, data_start_aw + Nsymbols*sps) % frame_samples).astype(int)
    data_samples_aw = rx_awgn[data_indices_aw]
    data_mat_aw = data_samples_aw.reshape((sps, Nsymbols), order='F')
    symbol_samples_aw = np.sum(data_mat_aw, axis=0).astype(np.complex128)
    print_snip('symbol_samples_aw (first)', symbol_samples_aw[:min(12, symbol_samples_aw.size)])

    rx_syms_before_aw = pskdemod(symbol_samples_aw, M)
    errs_before_aw, ser_before_aw = biterr(data_bits, rx_syms_before_aw)
    print(f"INFO: [DEM_BEFORE_AW] NumErr={errs_before_aw}, SER={ser_before_aw:.6g}")

    symbol_samples_eq_aw = symbol_samples_aw / h_est_aw
    print_snip('symbol_samples_eq_aw (first)', symbol_samples_eq_aw[:min(12, symbol_samples_eq_aw.size)])
    
    rx_syms_after_aw = pskdemod(symbol_samples_eq_aw, M)
    errs_after_aw, ser_after_aw = biterr(data_bits, rx_syms_after_aw)
    print(f"INFO: [DEM_AFTER_AW] NumErr={errs_after_aw}, SER={ser_after_aw:.6g}")
    print(f"INFO: [SUMMARY] NO_AWGN: BER_before={ser_before_no:.6g} BER_after={ser_after_no:.6g} | RECEIVED: BER_before={ser_before_aw:.6g} BER_after={ser_after_aw:.6g}")

    # Prepare data for plotting
    n_plot = min(20, data_symbols.size, symbol_samples_eq_aw.size)
    tx_plot = data_symbols[:n_plot]
    rx_plot = symbol_samples_eq_aw[:n_plot]
    
    return {
        'symbol_samples_aw': symbol_samples_aw,
        'symbol_samples_eq_aw': symbol_samples_eq_aw,
        'corr_no': corr_no,
        'corr_aw': corr_aw,
        'tx_plot': tx_plot,
        'rx_plot': rx_plot,
    }


def plot_single_frame_results(data):
    """Takes the results from run_single_frame_demo and plots them."""
    # Unpack data
    symbol_samples_aw = data['symbol_samples_aw']
    symbol_samples_eq_aw = data['symbol_samples_eq_aw']
    corr_no = data['corr_no']
    corr_aw = data['corr_aw']
    tx_plot = data['tx_plot']
    rx_plot = data['rx_plot']
    n_plot = len(tx_plot)

    # Plot 1: Before vs After Equalization
    plt.figure('Received Constellation: Before vs After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_aw[:2000]), np.imag(symbol_samples_aw[:2000]), '.')
    plt.title('Before Equalization (subset)')
    plt.xlabel('I'); plt.ylabel('Q'); plt.axis('equal'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_aw), np.imag(symbol_samples_eq_aw), '.')
    plt.title('After Equalization (subset)')
    plt.xlabel('I'); plt.ylabel('Q'); plt.axis('equal'); plt.grid(True)
    plt.suptitle('Received Signal Constellation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot 2: Correlation Peaks
    plt.figure('Correlation peaks', figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(corr_no); plt.title('Correlation magnitude (no AWGN)'); plt.ylabel('mag'); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(corr_aw); plt.title('Correlation magnitude (received buffer)'); plt.ylabel('mag'); plt.grid(True)
    
    # Plot 3: Tx vs Rx Constellation
    plt.figure('Tx vs Rx Constellation (first symbols)', figsize=(6, 6))
    plt.plot(np.real(tx_plot), np.imag(tx_plot), 'x', markersize=8, label=f'Transmitted (first {n_plot})')
    plt.plot(np.real(rx_plot), np.imag(rx_plot), 'o', markersize=6, label=f'Received EQ (first {n_plot})')
    for i in range(n_plot):
        plt.plot([np.real(tx_plot[i]), np.real(rx_plot[i])], [np.imag(tx_plot[i]), np.imag(rx_plot[i])], '-', linewidth=0.7, alpha=0.6)
    plt.title(f'Transmitted (x) vs Received after EQ (o) — first {n_plot} symbols')
    plt.xlabel('I'); plt.ylabel('Q'); plt.axis('equal'); plt.grid(True); plt.legend(loc='best')
    
    # Plot 4: Tx vs Rx time series
    inds = np.arange(n_plot)
    plt.figure('Tx vs Rx time series (first symbols)', figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(inds, np.real(tx_plot), 'x-', label='Tx real')
    plt.plot(inds, np.real(rx_plot), 'o-', label='Rx eq real')
    plt.ylabel('Real'); plt.grid(True); plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(inds, np.imag(tx_plot), 'x-', label='Tx imag')
    plt.plot(inds, np.imag(rx_plot), 'o-', label='Rx eq imag')
    plt.xlabel('Symbol index (0 = first data symbol)'); plt.ylabel('Imag'); plt.grid(True); plt.legend(loc='best')
    plt.tight_layout()


# ------------------- Monte Carlo helpers (Pluto-only) ---------------------
def simulate_frame_ser(M=16, sps=8, fs=1e6, Nsymbols=4000,
                       sync_len_bits=26, ch_pilot_len_bits=128,
                       rng=None, pluto_ip=PLUTO_IP):
    """Transmit a single random frame using Pluto and return (num_errors, Nsymbols).

    Note: This function does not perform any software AWGN injection — the
    received buffer is the real RF capture from the Pluto device.
    """
    if rng is None:
        rng = np.random

    # Build frame pieces (random symbols)
    sync_bits = _make_sync_bits(sync_len_bits)
    chpilot_bits = rng.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = rng.randint(0, M, size=(Nsymbols,))

    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nsymbols
    frame_samples = frame_bits * sps

    # Transmit and receive using Pluto
    pluto = PlutoSDRWrapper(ip=pluto_ip, sample_rate=fs, center_freq=GUI_CENTER_FREQ, rx_buffer_size=frame_samples)
    pluto.set_tx_gain(GUI_TX_GAIN)
    pluto.set_rx_gain(GUI_RX_GAIN, mode=GUI_RX_GAIN_MODE)

    tx_samples = (tx_frame * (2**14)).astype(np.complex64)
    pluto.tx_waveform(tx_samples, cyclic=True)
    pluto.rx_flush(n=5)
    rx = pluto.rx_once().astype(np.complex128)
    pluto.stop_tx()
    pluto.close()

    # Sync detect & pilot extraction
    rx2 = np.concatenate([rx, rx])
    corr = np.abs(np.convolve(rx2, np.conjugate(sync_shaped[::-1])))
    peak = np.argmax(corr)
    pilot_start_in_rx2 = peak - (len(sync_shaped) - 1)
    pilot_start = pilot_start_in_rx2 % frame_samples

    chpilot_start = pilot_start + len(sync_shaped)
    chpilot_indices = (np.arange(chpilot_start, chpilot_start + ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples = rx[chpilot_indices]
    chpilot_mat = chpilot_samples.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols = np.sum(chpilot_mat, axis=0).astype(np.complex128)

    h_est = np.mean(rx_chpilot_symbols / chpilot_symbols)

    # data extraction
    data_start = chpilot_start + ch_pilot_len_bits*sps
    data_indices = (np.arange(data_start, data_start + Nsymbols*sps) % frame_samples).astype(int)
    data_samples = rx[data_indices]
    data_mat = data_samples.reshape((sps, Nsymbols), order='F')
    symbol_samples = np.sum(data_mat, axis=0).astype(np.complex128)

    # equalize, demodulate & count errors
    symbol_samples_eq = symbol_samples / h_est
    rx_syms = pskdemod(symbol_samples_eq, M)
    num_errors = np.sum(rx_syms != data_bits)
    return int(num_errors), int(Nsymbols)


def monte_carlo_ser(n_trials=100, seed=None, **simulate_kwargs):
    """Run n_trials independent Pluto frame trials and estimate SER.

    Returns tuple (ser, total_errors, total_symbols).
    """
    if seed is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    total_err = 0
    total_sym = 0
    for _ in range(n_trials):
        err, sym = simulate_frame_ser(rng=rng, **simulate_kwargs)
        total_err += err
        total_sym += sym
    ser = total_err / total_sym if total_sym > 0 else np.nan
    return ser, total_err, total_sym


# ------------------------ Simulation & Plotting (Refactored) -----------------------

def run_sps_sweep(fs, M, Nsymbols, sps_list, ch_pilot_len_bits, sync_len_bits, n_trials):
    """Runs the SER vs SPS simulation and returns data for plotting."""
    sers = []
    for sps in sps_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials, M=M, sps=sps, fs=fs, Nsymbols=Nsymbols,
                                    ch_pilot_len_bits=ch_pilot_len_bits, sync_len_bits=sync_len_bits)
        print(f"INFO: sps={sps} SER={ser:.4e}")
        sers.append(ser)
    sps_arr = np.array(sps_list)
    Rs_arr = fs / sps_arr
    return {'sps_arr': sps_arr, 'sers': np.array(sers), 'Rs_arr': Rs_arr,
            'fs': fs, 'M': M, 'Nsymbols': Nsymbols}

def plot_sps_results(data):
    """Plots the data from the SER vs SPS simulation."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(data['sps_arr'], data['sers'], '-o')
    plt.title(f"SER vs sps (fs={data['fs']}, M={data['M']}, N={data['Nsymbols']})")
    plt.xlabel('samples per symbol (sps)'); plt.ylabel('SER (log scale)'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.semilogy(data['Rs_arr'], data['sers'], '-o')
    plt.title('SER vs Symbol Rate Rs')
    plt.xlabel('Rs = fs / sps (symbols/sec)'); plt.ylabel('SER (log scale)'); plt.grid(True)
    plt.tight_layout()

def run_sync_len_sweep(fs, M, Nsymbols, sync_len_list, sps, ch_pilot_len_bits, n_trials):
    sers = []
    for sync_len in sync_len_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials, M=M, sps=sps, fs=fs, Nsymbols=Nsymbols,
                                    ch_pilot_len_bits=ch_pilot_len_bits, sync_len_bits=sync_len)
        print(f"INFO: sync_len={sync_len} SER={ser:.4e}")
        sers.append(ser)
    return {'sync_len_list': np.array(sync_len_list), 'sers': np.array(sers), 'fs': fs, 'M': M,
            'sps': sps, 'Nsymbols': Nsymbols}

def plot_sync_len_results(data):
    plt.figure(figsize=(7, 5))
    plt.semilogy(data['sync_len_list'], data['sers'], '-o')
    plt.title(f"SER vs sync length (fs={data['fs']}, M={data['M']}, sps={data['sps']}, N={data['Nsymbols']})")
    plt.xlabel('sync / pilot length (symbols)'); plt.ylabel('SER (log scale)'); plt.grid(True)

def run_M_sweep(fs, M_list, Nsymbols, sps, ch_pilot_len_bits, sync_len_bits, n_trials):
    sers = []
    for M in M_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials, M=M, sps=sps, fs=fs, Nsymbols=Nsymbols,
                                    ch_pilot_len_bits=ch_pilot_len_bits, sync_len_bits=sync_len_bits)
        print(f"INFO: M={M} SER={ser:.4e}")
        sers.append(ser)
    return {'M_list': np.array(M_list), 'sers': np.array(sers), 'fs': fs, 'Nsymbols': Nsymbols, 'sps': sps}

def plot_M_results(data):
    plt.figure(figsize=(7, 5))
    plt.semilogy(data['M_list'], data['sers'], '-o')
    plt.title(f"SER vs Modulation order M (fs={data['fs']}, N={data['Nsymbols']}, sps={data['sps']})")
    plt.xlabel('M (PSK order)'); plt.ylabel('SER (log scale)'); plt.grid(True)

def run_N_sweep(fs, M, sps, N_list, ch_pilot_len_bits, sync_len_bits, n_trials):
    sers = []
    for Nsymbols in N_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials, M=M, sps=sps, fs=fs, Nsymbols=Nsymbols,
                                    ch_pilot_len_bits=ch_pilot_len_bits, sync_len_bits=sync_len_bits)
        print(f"INFO: N={Nsymbols} SER={ser:.4e}")
        sers.append(ser)
    return {'N_list': np.array(N_list), 'sers': np.array(sers), 'fs': fs, 'M': M, 'sps': sps}

def plot_N_results(data):
    plt.figure(figsize=(7, 5))
    plt.semilogy(data['N_list'], data['sers'], '-o')
    plt.title(f"SER vs Number of data symbols N (fs={data['fs']}, M={data['M']}, sps={data['sps']})")
    plt.xlabel('Number of data symbols (N)'); plt.ylabel('SER (log scale)'); plt.grid(True)


def find_best_params_for_M(M_list, sps_candidates, N_candidates, fs_candidates,
                           ch_pilot_len_bits=128, sync_len_bits=26,
                           n_trials=60, top_k=5):
    """Pluto-only coarse search across (sps, N, fs) for each M.

    Returns dict: M -> list of (ser, sps, N, fs, Rs)
    """
    results = {}
    for M in M_list:
        combos = []
        for sps in sps_candidates:
            for N in N_candidates:
                for fs in fs_candidates:
                    ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                                M=M, sps=sps, fs=fs, Nsymbols=N,
                                                ch_pilot_len_bits=ch_pilot_len_bits,
                                                sync_len_bits=sync_len_bits)
                    Rs = fs / sps
                    combos.append((ser, sps, N, fs, Rs))
                    print(f"INFO: M={M} sps={sps} N={N} fs={fs:.2e} Rs={Rs:.2f} SER={ser:.4e}")
        combos.sort(key=lambda x: x[0])
        results[M] = combos[:top_k]
    return results


# ------------------------------ Main guard --------------------------------
def main():
    # Instead of running everything automatically, present a simple GUI
    # so the user can choose which sweeps/demos to run and tweak their
    # parameters (prefilled with the function signature defaults).
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter import scrolledtext
    import threading

    def parse_list(s, cast=int):
        """Parse a comma/space/semicolon separated list into Python list with casting.
        Empty string -> empty list.
        """
        if s is None:
            return []
        s = str(s).strip()
        if s == "":
            return []
        # allow both commas and whitespace
        parts = []
        for p in s.replace(';', ',').split(','):
            p = p.strip()
            if not p:
                continue
            # if contains whitespace-separated values inside a comma token
            if any(c.isspace() for c in p) and ',' not in p:
                for q in p.split():
                    if q:
                        parts.append(q)
            else:
                parts.append(p)
        out = []
        for p in parts:
            try:
                out.append(cast(p))
            except Exception:
                # try float->cast int if needed
                try:
                    val = float(p)
                    out.append(cast(val))
                except Exception:
                    raise ValueError(f'Could not parse list element: {p}')
        return out

    def parse_list_float(s):
        if s is None:
            return []
        s = str(s).strip()
        if s == "":
            return []
        parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip()]
        return [float(p) for p in parts]

    def parse_sync_barker(s):
        # Accept 'None' or 'default' to indicate using default; otherwise parse ints
        if s is None:
            return None
        ss = str(s).strip()
        if ss == '' or ss.lower() in ('none', 'default'):
            return None
        parts = [p.strip() for p in ss.replace(';', ',').split(',') if p.strip()]
        arr = []
        for p in parts:
            try:
                arr.append(int(p))
            except Exception:
                raise ValueError(f'Invalid sync barker element: {p}')
        return np.array(arr, dtype=int)

    root = tk.Tk()
    root.title('Pluto M-PSK Demo — Select Runs & Parameters')

    mainframe = ttk.Frame(root, padding=10)
    mainframe.grid(row=0, column=0, sticky='nsew')

    # Configure resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Section helper
    def section(title, row):
        lbl = ttk.Label(mainframe, text=title, font=(None, 10, 'bold'))
        lbl.grid(row=row, column=0, sticky='w', pady=(8, 2), columnspan=6)
        return row + 1

    r = 0
    r = section('Select which routines to run (tick and edit parameters as needed)', r)

    # --- Global Pluto controls (applies to all runs) ---
    ttk.Label(mainframe, text='tx_gain (dB):').grid(row=r, column=0, sticky='e')
    global_tx_gain_entry = ttk.Entry(mainframe, width=10)
    global_tx_gain_entry.insert(0, str(int(GUI_TX_GAIN)))
    global_tx_gain_entry.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='Pluto center freq (MHz):').grid(row=r, column=2, sticky='e')
    global_cf_entry = ttk.Entry(mainframe, width=12)
    global_cf_entry.insert(0, str(int(GUI_CENTER_FREQ/1e6)))
    global_cf_entry.grid(row=r, column=3, columnspan=1, sticky='w')
    r += 1

    # --- RX Gain Mode Selection ---
    ttk.Label(mainframe, text='rx_gain_mode:').grid(row=r, column=0, sticky='e')
    gain_mode_var = tk.StringVar(value=GUI_RX_GAIN_MODE)
    rx_gain_mode_cb = ttk.Combobox(mainframe, textvariable=gain_mode_var,
                                   values=('manual', 'slow_attack', 'fast_attack'),
                                   state='readonly', width=12)
    rx_gain_mode_cb.grid(row=r, column=1, sticky='w')
    
    ttk.Label(mainframe, text='rx_gain (dB):').grid(row=r, column=2, sticky='e')
    global_rx_gain_entry = ttk.Entry(mainframe, width=12)
    global_rx_gain_entry.insert(0, str(int(GUI_RX_GAIN)))
    global_rx_gain_entry.grid(row=r, column=3, sticky='w')

    def on_gain_mode_change(event=None):
        if gain_mode_var.get() == 'manual':
            global_rx_gain_entry.config(state='normal')
        else:
            global_rx_gain_entry.config(state='disabled')

    rx_gain_mode_cb.bind('<<ComboboxSelected>>', on_gain_mode_change)
    on_gain_mode_change() # Set initial state
    r += 1


    # ------------------ run_single_frame_demo parameters ------------------
    run_demo_var = tk.IntVar(value=1) # Default to checked
    ttk.Checkbutton(mainframe, text='Run single-frame demo', variable=run_demo_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    # Defaults as in function signature
    demo_defaults = dict(Nsymbols=4000, M=16, sps=8, fs=1e6, sync_barker13='None', ch_pilot_len_bits=128, seed='None', pluto_ip=PLUTO_IP)

    ttk.Label(mainframe, text='Nsymbols:').grid(row=r, column=0, sticky='e')
    demo_Nbits = ttk.Entry(mainframe, width=12)
    demo_Nbits.insert(0, str(demo_defaults['Nsymbols']))
    demo_Nbits.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='M:').grid(row=r, column=2, sticky='e')
    demo_M = ttk.Entry(mainframe, width=8)
    demo_M.insert(0, str(demo_defaults['M']))
    demo_M.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='sps:').grid(row=r, column=4, sticky='e')
    demo_sps = ttk.Entry(mainframe, width=8)
    demo_sps.insert(0, str(demo_defaults['sps']))
    demo_sps.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='fs (MHz):').grid(row=r, column=0, sticky='e')
    demo_fs = ttk.Entry(mainframe, width=14)
    demo_fs.insert(0, str(int(demo_defaults['fs']/1e6)))
    demo_fs.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='sync_barker13 (None or comma list):').grid(row=r, column=2, sticky='e')
    demo_sync = ttk.Entry(mainframe, width=28)
    demo_sync.insert(0, demo_defaults['sync_barker13'])
    demo_sync.grid(row=r, column=3, columnspan=3, sticky='w')
    r += 1

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r, column=0, sticky='e')
    demo_chpilot = ttk.Entry(mainframe, width=12)
    demo_chpilot.insert(0, str(demo_defaults['ch_pilot_len_bits']))
    demo_chpilot.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='seed (int or None):').grid(row=r, column=2, sticky='e')
    demo_seed = ttk.Entry(mainframe, width=12)
    demo_seed.insert(0, demo_defaults['seed'])
    demo_seed.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='pluto_ip:').grid(row=r, column=4, sticky='e')
    demo_pluto_ip = ttk.Entry(mainframe, width=20)
    demo_pluto_ip.insert(0, demo_defaults['pluto_ip'])
    demo_pluto_ip.grid(row=r, column=5, sticky='w')
    r += 1

    # ------------------ plot_ser_vs_sps parameters ------------------
    run_sps_var = tk.IntVar(value=0)
    ttk.Checkbutton(mainframe, text='Plot SER vs sps', variable=run_sps_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    # Defaults from function signature
    sps_defaults = '1,2,4,8,16'
    ttk.Label(mainframe, text='fs (MHz):').grid(row=r, column=0, sticky='e')
    sps_fs = ttk.Entry(mainframe, width=14)
    sps_fs.insert(0, str(1))
    sps_fs.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='M:').grid(row=r, column=2, sticky='e')
    sps_M = ttk.Entry(mainframe, width=8)
    sps_M.insert(0, '16')
    sps_M.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='Nsymbols:').grid(row=r, column=4, sticky='e')
    sps_Nbits = ttk.Entry(mainframe, width=12)
    sps_Nbits.insert(0, '4000')
    sps_Nbits.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='sps_list (comma sep):').grid(row=r, column=0, sticky='e')
    sps_entry = ttk.Entry(mainframe, width=36)
    sps_entry.insert(0, sps_defaults)
    sps_entry.grid(row=r, column=1, columnspan=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r, column=0, sticky='e')
    sps_chpilot = ttk.Entry(mainframe, width=12)
    sps_chpilot.insert(0, '128')
    sps_chpilot.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='sync_len_bits:').grid(row=r, column=2, sticky='e')
    sps_synclen = ttk.Entry(mainframe, width=12)
    sps_synclen.insert(0, '26')
    sps_synclen.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='n_trials:').grid(row=r, column=4, sticky='e')
    sps_ntr = ttk.Entry(mainframe, width=8)
    sps_ntr.insert(0, '80')
    sps_ntr.grid(row=r, column=5, sticky='w')
    r += 1

    # ------------------ plot_ser_vs_sync_len parameters ------------------
    run_sync_var = tk.IntVar(value=0)
    ttk.Checkbutton(mainframe, text='Plot SER vs sync length', variable=run_sync_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    ttk.Label(mainframe, text='fs (MHz):').grid(row=r, column=0, sticky='e')
    sync_fs = ttk.Entry(mainframe, width=14)
    sync_fs.insert(0, str(1))
    sync_fs.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='M:').grid(row=r, column=2, sticky='e')
    sync_M = ttk.Entry(mainframe, width=8)
    sync_M.insert(0, '16')
    sync_M.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='Nsymbols:').grid(row=r, column=4, sticky='e')
    sync_Nbits = ttk.Entry(mainframe, width=12)
    sync_Nbits.insert(0, '4000')
    sync_Nbits.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='sync_len_list (comma sep):').grid(row=r, column=0, sticky='e')
    sync_entry = ttk.Entry(mainframe, width=36)
    sync_entry.insert(0, '8,13,26,52')
    sync_entry.grid(row=r, column=1, columnspan=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='sps:').grid(row=r, column=0, sticky='e')
    sync_sps = ttk.Entry(mainframe, width=8)
    sync_sps.insert(0, '8')
    sync_sps.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r, column=2, sticky='e')
    sync_chpilot = ttk.Entry(mainframe, width=12)
    sync_chpilot.insert(0, '128')
    sync_chpilot.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='n_trials:').grid(row=r, column=4, sticky='e')
    sync_ntr = ttk.Entry(mainframe, width=8)
    sync_ntr.insert(0, '80')
    sync_ntr.grid(row=r, column=5, sticky='w')
    r += 1

    # ------------------ plot_ser_vs_M parameters ------------------
    run_M_var = tk.IntVar(value=0)
    ttk.Checkbutton(mainframe, text='Plot SER vs modulation order M', variable=run_M_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    ttk.Label(mainframe, text='fs (MHz):').grid(row=r, column=0, sticky='e')
    M_fs = ttk.Entry(mainframe, width=14)
    M_fs.insert(0, str(1))
    M_fs.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='M list (comma sep):').grid(row=r, column=2, sticky='e')
    M_entry = ttk.Entry(mainframe, width=18)
    M_entry.insert(0, '2,4,8,16,32')
    M_entry.grid(row=r, column=3, columnspan=3, sticky='w')
    r += 1

    ttk.Label(mainframe, text='Nsymbols:').grid(row=r, column=0, sticky='e')
    M_Nbits = ttk.Entry(mainframe, width=12)
    M_Nbits.insert(0, '1000')
    M_Nbits.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='sps:').grid(row=r, column=2, sticky='e')
    M_sps = ttk.Entry(mainframe, width=8)
    M_sps.insert(0, '8')
    M_sps.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r, column=4, sticky='e')
    M_chpilot = ttk.Entry(mainframe, width=8)
    M_chpilot.insert(0, '128')
    M_chpilot.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='sync_len_bits:').grid(row=r, column=0, sticky='e')
    M_synclen = ttk.Entry(mainframe, width=12)
    M_synclen.insert(0, '26')
    M_synclen.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='n_trials:').grid(row=r, column=2, sticky='e')
    M_ntr = ttk.Entry(mainframe, width=8)
    M_ntr.insert(0, '80')
    M_ntr.grid(row=r, column=3, sticky='w')
    r += 1

    # ------------------ plot_ser_vs_N parameters ------------------
    run_N_var = tk.IntVar(value=0)
    ttk.Checkbutton(mainframe, text='Plot SER vs number of data symbols N', variable=run_N_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    ttk.Label(mainframe, text='fs (MHz):').grid(row=r, column=0, sticky='e')
    N_fs = ttk.Entry(mainframe, width=14)
    N_fs.insert(0, str(1))
    N_fs.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='M:').grid(row=r, column=2, sticky='e')
    N_M = ttk.Entry(mainframe, width=8)
    N_M.insert(0, '16')
    N_M.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='sps:').grid(row=r, column=4, sticky='e')
    N_sps = ttk.Entry(mainframe, width=8)
    N_sps.insert(0, '8')
    N_sps.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='N list (comma sep):').grid(row=r, column=0, sticky='e')
    N_entry = ttk.Entry(mainframe, width=36)
    N_entry.insert(0, '500,1000,2000,4000')
    N_entry.grid(row=r, column=1, columnspan=5, sticky='w')

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r+1, column=0, sticky='e')
    N_chpilot = ttk.Entry(mainframe, width=12)
    N_chpilot.insert(0, '128')
    N_chpilot.grid(row=r+1, column=1, sticky='w')

    ttk.Label(mainframe, text='sync_len_bits:').grid(row=r+1, column=2, sticky='e')
    N_synclen = ttk.Entry(mainframe, width=12)
    N_synclen.insert(0, '26')
    N_synclen.grid(row=r+1, column=3, sticky='w')

    ttk.Label(mainframe, text='n_trials:').grid(row=r+1, column=4, sticky='e')
    N_ntr = ttk.Entry(mainframe, width=8)
    N_ntr.insert(0, '80')
    N_ntr.grid(row=r+1, column=5, sticky='w')
    r += 2

    # ------------------ find_best_params_for_M parameters ------------------
    run_search_var = tk.IntVar(value=0)
    ttk.Checkbutton(mainframe, text='Run coarse search (find_best_params_for_M)', variable=run_search_var).grid(row=r, column=0, sticky='w', columnspan=2)
    r += 1

    ttk.Label(mainframe, text='M list (comma sep):').grid(row=r, column=0, sticky='e')
    search_M_entry = ttk.Entry(mainframe, width=18)
    search_M_entry.insert(0, '4,16')
    search_M_entry.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='sps candidates:').grid(row=r, column=2, sticky='e')
    search_sps_entry = ttk.Entry(mainframe, width=18)
    search_sps_entry.insert(0, '2,4,8')
    search_sps_entry.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='N candidates:').grid(row=r, column=4, sticky='e')
    search_N_entry = ttk.Entry(mainframe, width=18)
    search_N_entry.insert(0, '1000,2000')
    search_N_entry.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='fs candidates (MHz, comma sep):').grid(row=r, column=0, sticky='e')
    search_fs_entry = ttk.Entry(mainframe, width=28)
    search_fs_entry.insert(0, '1,2')
    search_fs_entry.grid(row=r, column=1, columnspan=3, sticky='w')

    ttk.Label(mainframe, text='ch_pilot_len_bits:').grid(row=r, column=4, sticky='e')
    search_chpilot = ttk.Entry(mainframe, width=10)
    search_chpilot.insert(0, '128')
    search_chpilot.grid(row=r, column=5, sticky='w')
    r += 1

    ttk.Label(mainframe, text='sync_len_bits:').grid(row=r, column=0, sticky='e')
    search_synclen = ttk.Entry(mainframe, width=10)
    search_synclen.insert(0, '26')
    search_synclen.grid(row=r, column=1, sticky='w')

    ttk.Label(mainframe, text='n_trials:').grid(row=r, column=2, sticky='e')
    search_ntr = ttk.Entry(mainframe, width=8)
    search_ntr.insert(0, '60')
    search_ntr.grid(row=r, column=3, sticky='w')

    ttk.Label(mainframe, text='top_k:').grid(row=r, column=4, sticky='e')
    search_topk = ttk.Entry(mainframe, width=8)
    search_topk.insert(0, '5')
    search_topk.grid(row=r, column=5, sticky='w')
    r += 1

    # Run button
    def run_selected():
        # --- FIX: Use a queue to pass plot data back to main thread ---
        results_queue = queue.Queue()
        
        # Disable run button to prevent multiple clicks
        run_btn.config(state='disabled')

        output_win = tk.Toplevel(root)
        output_win.title('Run Output')
        txt = scrolledtext.ScrolledText(output_win, wrap='word', width=100, height=30)
        txt.pack(fill='both', expand=True)
        log_queue = queue.Queue()

        def poll_log_queue():
            try:
                while True:
                    s = log_queue.get_nowait()
                    txt.insert('end', s)
                    txt.see('end')
            except queue.Empty:
                pass
            if output_win.winfo_exists():
                output_win.after(100, poll_log_queue)
        poll_log_queue()

        def worker():
            import builtins, io, traceback
            original_print = builtins.print
            def gui_print(*args, **kwargs):
                original_print(*args, **kwargs)
                output_capture = io.StringIO()
                gui_kwargs = kwargs.copy()
                gui_kwargs['file'] = output_capture
                original_print(*args, **gui_kwargs)
                log_queue.put(output_capture.getvalue())
            builtins.print = gui_print
            
            plot_data = {}
            try:
                globals()['GUI_TX_GAIN'] = float(global_tx_gain_entry.get())
                globals()['GUI_RX_GAIN'] = float(global_rx_gain_entry.get())
                globals()['GUI_RX_GAIN_MODE'] = gain_mode_var.get()
                globals()['GUI_CENTER_FREQ'] = float(global_cf_entry.get()) * 1e6
                
                if run_demo_var.get():
                    # ... parameter parsing ...
                    params = {
                        'Nsymbols': int(demo_Nbits.get()), 'M': int(demo_M.get()), 'sps': int(demo_sps.get()),
                        'fs': float(demo_fs.get()) * 1e6, 'sync_barker13': parse_sync_barker(demo_sync.get()),
                        'ch_pilot_len_bits': int(demo_chpilot.get()),
                        'seed': None if demo_seed.get().strip().lower() in ('none', '') else int(demo_seed.get().strip()),
                        'pluto_ip': demo_pluto_ip.get().strip() or PLUTO_IP
                    }
                    print(f'Running single-frame demo with {params}')
                    plot_data['single_frame'] = run_single_frame_demo(**params)

                if run_sps_var.get():
                    params = {
                        'fs': float(sps_fs.get()) * 1e6, 'M': int(sps_M.get()), 'Nsymbols': int(sps_Nbits.get()),
                        'sps_list': parse_list(sps_entry.get(), int), 'ch_pilot_len_bits': int(sps_chpilot.get()),
                        'sync_len_bits': int(sps_synclen.get()), 'n_trials': int(sps_ntr.get())
                    }
                    print(f'Running SER vs SPS sweep with {params}')
                    plot_data['sps_sweep'] = run_sps_sweep(**params)

                if run_sync_var.get():
                    params = {
                        'fs': float(sync_fs.get()) * 1e6, 'M': int(sync_M.get()), 'Nsymbols': int(sync_Nbits.get()),
                        'sync_len_list': parse_list(sync_entry.get(), int), 'sps': int(sync_sps.get()),
                        'ch_pilot_len_bits': int(sync_chpilot.get()), 'n_trials': int(sync_ntr.get())
                    }
                    print(f'Running SER vs Sync Length sweep with {params}')
                    plot_data['sync_len_sweep'] = run_sync_len_sweep(**params)

                if run_M_var.get():
                    params = {
                        'fs': float(M_fs.get()) * 1e6, 'M_list': parse_list(M_entry.get(), int),
                        'Nsymbols': int(M_Nbits.get()), 'sps': int(M_sps.get()),
                        'ch_pilot_len_bits': int(M_chpilot.get()), 'sync_len_bits': int(M_synclen.get()),
                        'n_trials': int(M_ntr.get())
                    }
                    print(f'Running SER vs M sweep with {params}')
                    plot_data['m_sweep'] = run_M_sweep(**params)

                if run_N_var.get():
                    params = {
                        'fs': float(N_fs.get()) * 1e6, 'M': int(N_M.get()), 'sps': int(N_sps.get()),
                        'N_list': parse_list(N_entry.get(), int), 'ch_pilot_len_bits': int(N_chpilot.get()),
                        'sync_len_bits': int(N_synclen.get()), 'n_trials': int(N_ntr.get())
                    }
                    print(f'Running SER vs N sweep with {params}')
                    plot_data['n_sweep'] = run_N_sweep(**params)

                if run_search_var.get():
                    # This function doesn't produce a plottable result, just prints
                    params = {
                        'M_list': parse_list(search_M_entry.get(), int), 'sps_candidates': parse_list(search_sps_entry.get(), int),
                        'N_candidates': parse_list(search_N_entry.get(), int),
                        'fs_candidates': [f * 1e6 for f in parse_list_float(search_fs_entry.get())],
                        'ch_pilot_len_bits': int(search_chpilot.get()), 'sync_len_bits': int(search_synclen.get()),
                        'n_trials': int(search_ntr.get()), 'top_k': int(search_topk.get())
                    }
                    print(f'Running coarse search with {params}')
                    results = find_best_params_for_M(**params)
                    print('--- Coarse Search Results ---')
                    for Mkey, combos in results.items():
                        print(f'M={Mkey}')
                        for ser, sps_v, N_v, fs_v, Rs_v in combos:
                            print(f'  SER={ser:.4e} sps={sps_v} N={N_v} fs={fs_v:.2e} Rs={Rs_v:.2f}')
                    print('-----------------------------')
                
                results_queue.put(plot_data)

            except Exception as e:
                print(f"\n--- An error occurred ---\n{e}")
                print(traceback.format_exc())
                results_queue.put({'error': e}) # Put error in queue to notify main thread
            finally:
                builtins.print = original_print

        def check_for_results():
            try:
                results = results_queue.get_nowait()
                
                if 'error' in results:
                    messagebox.showerror('Error', f"An error occurred in the worker thread: {results['error']}")
                else:
                    # If we have any data, generate the plots
                    if not results:
                        messagebox.showinfo('No selection', 'No routines selected — nothing to run.')
                    else:
                        print('\nAll selected tasks finished. Generating plots...')
                        if 'single_frame' in results:
                            plot_single_frame_results(results['single_frame'])
                        if 'sps_sweep' in results:
                            plot_sps_results(results['sps_sweep'])
                        if 'sync_len_sweep' in results:
                            plot_sync_len_results(results['sync_len_sweep'])
                        if 'm_sweep' in results:
                            plot_M_results(results['m_sweep'])
                        if 'n_sweep' in results:
                            plot_N_results(results['n_sweep'])
                        
                        plt.show() # This is now safe to call
                
                run_btn.config(state='normal') # Re-enable the button

            except queue.Empty:
                # If the queue is empty, check again in 100ms
                root.after(100, check_for_results)

        # Start the worker thread and the poller
        threading.Thread(target=worker, daemon=True).start()
        root.after(100, check_for_results)

    run_btn = ttk.Button(mainframe, text='Run selected simulations', command=run_selected)
    run_btn.grid(row=r, column=0, columnspan=3, pady=(12, 0), sticky='w')

    quit_btn = ttk.Button(mainframe, text='Quit', command=root.destroy)
    quit_btn.grid(row=r, column=3, columnspan=3, pady=(12, 0), sticky='e')

    root.mainloop()
    
if __name__ == "__main__":
    main()
