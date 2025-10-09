#!/usr/bin/env python3
"""
Complete Python implementation ported from the provided MATLAB script.

Features:
- QAM and PSK modulation/demodulation with Gray mapping (unit average power for QAM).
- Four experiments:
    1) SER vs Samples-per-symbol (sps)
    2) SER vs Pilot length
    3) SER vs Modulation order M
    4) SER vs Number of data symbols N
- Optional PlutoSDR usage via pyadi-iio (auto-detected). If not present, a realistic simulated
  channel is used (flat fading + timing offset + AWGN).
- Plots produced with matplotlib (one figure per experiment).

Author: Converted for user request.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import log2, ceil, floor
import time
import sys

# Try to import Pluto SDR interface
USE_SDR = False
try:
    import adi
    USE_SDR = True
except Exception:
    USE_SDR = False
    # If you want to force SDR but import failed, set USE_SDR = False and install pyadi-iio.

# ---------------------------
# Helper functions
# ---------------------------

def int_to_gray(n):
    return n ^ (n >> 1)

def gray_to_int(g):
    # inverse of gray code
    n = g
    mask = n >> 1
    while mask != 0:
        n ^= mask
        mask >>= 1
    return n

def bits_to_integers(bits, k):
    """
    bits: 1D array of 0/1 length = N*k
    returns: integer array length N, left-msb mapping (first bit is MSB)
    """
    bits = np.asarray(bits).astype(int)
    if bits.size == 0:
        return np.array([], dtype=int)
    assert bits.size % k == 0
    bits = bits.reshape(-1, k)
    # left-msb: e.g. [b0 b1 ...] where b0 is MSB
    ints = bits.dot(1 << np.arange(k - 1, -1, -1))
    return ints

def integers_to_bits(ints, k):
    """
    ints: 1D integer array
    returns: bits (1D) with left-msb ordering per symbol
    """
    ints = np.asarray(ints).astype(int)
    if ints.size == 0:
        return np.array([], dtype=int)
    bits = (((ints[:, None] & (1 << np.arange(k - 1, -1, -1))) > 0).astype(int)).reshape(-1)
    return bits

def normalize_constellation(constellation):
    """Scale constellation to unit average power."""
    power = np.mean(np.abs(constellation) ** 2)
    return constellation / np.sqrt(power)

# QAM constellation builder that supports rectangular constellations (e.g., 8, 128, etc.)
def build_qam_constellation(M):
    """
    Build Gray-coded rectangular QAM constellation of size M (power of 2).
    Returns array of complex symbols indexed by integer symbol value (0..M-1).
    Mapping: integer -> Gray-coded (2D).
    """
    k = int(log2(M))
    # choose number of levels on I and Q
    # distribute bits: bits_i = ceil(k/2), bits_q = floor(k/2)
    bits_i = ceil(k / 2)
    bits_q = k - bits_i
    Mi = 2 ** bits_i
    Mq = 2 ** bits_q
    # Create amplitude levels (odd integers centered on 0): e.g., for 4 levels -> [-3, -1, +1, +3]
    def levels(m):
        # m levels => values: -(m-1), -(m-3), ..., (m-1)
        vals = np.arange(m) * 2 - (m - 1)
        return vals
    levels_i = levels(Mi)
    levels_q = levels(Mq)
    # Normalize later
    constellation = np.zeros(M, dtype=complex)
    # Gray mapping on each axis
    for idx in range(M):
        # split idx into axis integer indexes using binary extraction: lower bits -> Q-axis
        # We'll map axis indices by Gray code so adjacent integers differ by 1 bit per axis.
        # Represent idx as integer 0..M-1, split into bits_i MSBs for I and bits_q LSBs for Q.
        i_index = idx >> bits_q
        q_index = idx & ((1 << bits_q) - 1) if bits_q > 0 else 0
        # convert to Gray-coded indices
        i_gray = int_to_gray(i_index)
        q_gray = int_to_gray(q_index)
        # map to amplitude levels
        # clamp index range just in case
        i_level = levels_i[i_gray % Mi]
        q_level = levels_q[q_gray % Mq]
        constellation[idx] = i_level + 1j * q_level
    # normalize to unit average power
    constellation = normalize_constellation(constellation)
    return constellation

def build_psk_constellation(M):
    """
    M-PSK Gray mapping: we map integers 0..M-1 to phases, but use integer->Gray->phase such that
    successive Gray integer labels differ by one bit; mapping is symbol = exp(j*2*pi*gray/M).
    """
    constellation = np.zeros(M, dtype=complex)
    for i in range(M):
        g = int_to_gray(i)
        constellation[i] = np.exp(1j * 2 * np.pi * g / M)
    constellation = normalize_constellation(constellation)
    return constellation

def modulate_bits(bits, M, mod_type='QAM'):
    """
    bits -> complex symbols using Gray mapping.
    bits: 1D array of 0/1 length = k * N
    returns: complex array of length N
    """
    k = int(log2(M))
    ints = bits_to_integers(bits, k)
    if mod_type.upper() == 'QAM':
        C = build_qam_constellation(M)
    else:
        C = build_psk_constellation(M)
    return C[ints]

def demodulate_symbols(symbols, M, mod_type='QAM'):
    """
    symbols: complex array (length N)
    returns: bits (1D array length = k * N) in left-msb ordering
    """
    if len(symbols) == 0:
        return np.array([], dtype=int)
    if mod_type.upper() == 'QAM':
        C = build_qam_constellation(M)
    else:
        C = build_psk_constellation(M)
    # compute nearest neighbor
    # distances: NxM
    dists = np.abs(symbols.reshape(-1, 1) - C.reshape(1, -1))
    idx = np.argmin(dists, axis=1)
    k = int(log2(M))
    bits = integers_to_bits(idx, k)
    return bits

def symbol_error_rate(tx_ints, rx_ints):
    """
    tx_ints, rx_ints: integer arrays same length
    returns: (num_errors, ser)
    """
    tx_ints = np.asarray(tx_ints)
    rx_ints = np.asarray(rx_ints)
    if tx_ints.size == 0:
        return 0, 0.0
    assert tx_ints.size == rx_ints.size
    errs = np.sum(tx_ints != rx_ints)
    return int(errs), float(errs) / tx_ints.size

def circ_extract(vec, start_idx, L):
    """
    MATLAB-style circular extract:
    start_idx is 0-based here. Returns length L array from vec with wrap-around.
    """
    n = len(vec)
    idxs = (np.arange(start_idx, start_idx + L) % n).astype(int)
    return vec[idxs]

# ---------------------------
# SDR wrapper (Pluto) or simulator
# ---------------------------

class SDRInterfaceSimulator:
    """
    Simulates a channel: applies flat complex gain (random), optional delay, and AWGN.
    Useful when no hardware available.
    """

    def __init__(self, fs, fc, rx_gain_db=30, tx_gain_db=-10, buffer_size=1024, snr_db=30):
        self.fs = fs
        self.fc = fc
        self.rx_gain_db = rx_gain_db
        self.tx_gain_db = tx_gain_db
        self.buffer_size = buffer_size
        self.snr_db = snr_db
        # keep a copy of last tx buffer for correlation
        self.last_tx = np.zeros(buffer_size, dtype=complex)

    def start_tx(self, tx_buffer):
        # In real hardware you'd call continuous transmit. We store tx_buffer for simulation
        self.last_tx = np.asarray(tx_buffer).astype(complex)

    def rx(self):
        """
        Return simulated received_data array same length as stored tx buffer:
        - apply random flat fading scalar h
        - add random timing offset (circular shift)
        - add AWGN according to snr_db
        """
        tx = self.last_tx.copy()
        if tx.size == 0:
            return np.zeros(self.buffer_size, dtype=complex)

        # pick a random channel scalar close to 1 plus small random phase
        h = (1.0 + 0.0j) * (0.8 + 0.4j)  # deterministic-ish for repeatable results if desired
        # We'll add a small random perturbation
        h *= (1 + 0.05 * (np.random.randn() + 1j * np.random.randn()))

        rx = tx * h

        # Add AWGN noise: compute noise variance from desired SNR
        signal_power = np.mean(np.abs(rx) ** 2) if rx.size > 0 else 1.0
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-12)
        noise = (np.sqrt(noise_power / 2) *
                 (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape)))
        rx += noise

        # Add a small random circular delay to mimic timing
        delay = np.random.randint(0, min(200, rx.size)) if rx.size > 10 else 0
        if delay != 0:
            rx = np.concatenate((rx[delay:], rx[:delay]))

        # Truncate/pad to buffer_size
        if rx.size < self.buffer_size:
            rx = np.pad(rx, (0, self.buffer_size - rx.size))
        elif rx.size > self.buffer_size:
            rx = rx[:self.buffer_size]
        return rx

# optional real Pluto interface (thin wrapper)
class PlutoWrapper:
    def __init__(self, uri, fs, fc, tx_gain, rx_gain, buffer_size):
        self.uri = uri
        self.fs = int(fs)
        self.fc = int(fc)
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.buffer_size = int(buffer_size)
        # create device
        self.dev = adi.Pluto(uri)
        # configure tx
        self.dev.tx_sample_rate = self.fs
        self.dev.tx_lo = self.fc
        self.dev.tx_hardwaregain_chan0 = int(self.tx_gain)
        # configure rx
        self.dev.rx_sample_rate = self.fs
        self.dev.rx_lo = self.fc
        try:
            # interface attribute names vary by pyadi-iio versions; set a few likely ones
            self.dev.rx_buffer_size = int(self.buffer_size)
        except Exception:
            pass
        try:
            self.dev.rx_hardwaregain_chan0 = int(self.rx_gain)
        except Exception:
            pass
        self.last_tx = np.zeros(self.buffer_size, dtype=np.complex64)

    def start_tx(self, tx_buffer):
        # Ensure correct dtype
        tx_buffer = np.asarray(tx_buffer).astype(np.complex64)
        # pyadi-iio expects interleaved I/Q or complex float? For pyadi-iio usually complex64 is OK.
        # We'll call the tx() method to push one buffer (device-specific behavior).
        try:
            self.dev.tx(tx_buffer)  # send once
        except Exception:
            # fallback: save buffer and skip hardware transmission
            pass
        self.last_tx = tx_buffer

    def rx(self):
        # Attempt to read samples
        try:
            data = self.dev.rx()  # may return complex numpy array
            data = np.asarray(data).astype(complex)
            # ensure length buffer_size
            if data.size < self.buffer_size:
                data = np.pad(data, (0, self.buffer_size - data.size))
            elif data.size > self.buffer_size:
                data = data[:self.buffer_size]
            return data
        except Exception:
            # fallback to zeros or last tx echoed
            return np.zeros(self.buffer_size, dtype=complex)

# ---------------------------
# Main experiment function
# ---------------------------

def run_experiments(modulation_type='QAM'):
    # ===================== Basic Parameters =====================
    modulation_type = modulation_type.upper()
    fs = 2e6
    fc = 915e6
    tx_gain = -10
    rx_gain = 30

    max_sps = 64
    max_N = 4096
    max_pilot = 512

    if modulation_type == 'QAM':
        max_k = int(log2(256))
    else:
        max_k = int(log2(16))

    max_symbols = max_N + max_pilot
    max_samples = max_symbols * max_sps
    buffer_size = 4 * max_samples

    print(f"Buffer size: {buffer_size}")

    # instantiate SDR or simulator
    if USE_SDR:
        try:
            print("Attempting to initialize PlutoSDR (pyadi-iio)...")
            sdr = PlutoWrapper(uri="ip:192.168.2.1", fs=fs, fc=fc,
                               tx_gain=tx_gain, rx_gain=rx_gain,
                               buffer_size=buffer_size)
            mode_desc = "PlutoSDR (hardware)"
        except Exception as e:
            print("Pluto init failed, falling back to simulator:", e)
            sdr = SDRInterfaceSimulator(fs, fc, rx_gain_db=rx_gain,
                                        tx_gain_db=tx_gain, buffer_size=buffer_size,
                                        snr_db=30)
            mode_desc = "Simulator"
    else:
        print("pyadi-iio not installed or not found — using simulator.")
        sdr = SDRInterfaceSimulator(fs, fc, rx_gain_db=rx_gain,
                                    tx_gain_db=tx_gain, buffer_size=buffer_size,
                                    snr_db=30)
        mode_desc = "Simulator"

    print("Running in mode:", mode_desc)

    # functions set based on modulation
    if modulation_type == 'PSK':
        mod_func = lambda bits, M: modulate_bits(bits, M, mod_type='PSK')
        demod_func = lambda syms, M: demodulate_symbols(syms, M, mod_type='PSK')
        M_vec = [2, 4, 8, 16]
    elif modulation_type == 'QAM':
        mod_func = lambda bits, M: modulate_bits(bits, M, mod_type='QAM')
        demod_func = lambda syms, M: demodulate_symbols(syms, M, mod_type='QAM')
        M_vec = [4, 8, 16, 64, 128, 256]
    else:
        raise ValueError("Invalid modulation_type")

    # Transmission buffer (complex baseband)
    tx_buffer = np.zeros(buffer_size, dtype=complex)

    # -----------------------
    # 1) SER vs sps
    # -----------------------
    print('\n--- Plot 1: SER vs. Samples per Symbol (sps) ---')
    sps_vec = [2, 4, 8, 16, 32, 64]
    ser_vs_sps = np.zeros(len(sps_vec))
    N_const = 2048
    M_const = M_vec[1]  # pick second element as in MATALB (index 2 in MATLAB)
    pilot_len_const = 128
    k_const = int(log2(M_const))

    # Start continuous TX (simulator/hardware wrapper)
    sdr.start_tx(tx_buffer)

    for i, sps in enumerate(sps_vec):
        print(f"Testing sps = {sps} ...")
        pilot_bits = np.random.randint(0, 2, pilot_len_const * k_const)
        data_bits = np.random.randint(0, 2, N_const * k_const)

        tx_pilot_syms = mod_func(pilot_bits, M_const)
        tx_data_syms = mod_func(data_bits, M_const)
        tx_syms = np.concatenate((tx_pilot_syms, tx_data_syms))

        # Upsample
        pilot_upsampled = np.repeat(tx_pilot_syms, sps)
        tx_signal = np.repeat(tx_syms, sps)

        # load into tx_buffer and start
        tx_buffer[:] = 0
        tx_buffer[:len(tx_signal)] = tx_signal
        sdr.start_tx(tx_buffer)

        time.sleep(0.3)  # allow some time for TX/propagation (short)

        received_data = sdr.rx()

        # Synchronization using pilot correlation
        # correlate returns length = len(received) - len(pilot_upsampled) + 1 valid positions if mode='valid'
        corr = np.abs(correlate(received_data, np.conj(pilot_upsampled), mode='valid'))
        if corr.size == 0:
            peak_idx = 0
        else:
            peak_idx = np.argmax(corr)
        # start_idx in MATLAB code was peak_idx - length(pilot_upsampled) + 1 (1-based)
        # For our 'valid' cross-correlation definition, start_idx is simply peak_idx
        start_idx = int(peak_idx)
        # extract aligned frame (circular)
        rx_aligned = circ_extract(received_data, start_idx, len(tx_signal))
        rx_down = rx_aligned[::sps]

        # Channel estimation using pilot symbols
        rx_pilot_down = rx_down[:pilot_len_const]
        # protect division
        with np.errstate(divide='ignore', invalid='ignore'):
            h_est = np.mean(rx_pilot_down / tx_pilot_syms)
            if np.abs(h_est) == 0 or np.isnan(h_est):
                h_est = 1.0 + 0j
        rx_equalized = rx_down / h_est

        # Demodulate
        rx_bits = demod_func(rx_equalized, M_const)
        rx_data_bits = rx_bits[pilot_len_const * k_const:]

        # Symbol integer conversion (left-msb)
        tx_symbols_int = bits_to_integers(data_bits, k_const)
        rx_symbols_int = bits_to_integers(rx_data_bits, k_const)

        # match lengths
        minlen = min(len(tx_symbols_int), len(rx_symbols_int))
        if minlen == 0:
            errs, ser = 0, 1.0
        else:
            errs, ser = symbol_error_rate(tx_symbols_int[:minlen], rx_symbols_int[:minlen])
        ser_vs_sps[i] = ser
        print(f"  SER = {ser:.6f} (errors={errs}/{minlen})")

    plt.figure()
    plt.semilogy(sps_vec, ser_vs_sps, 'bo-')
    plt.title(f'SER vs Samples per Symbol for {modulation_type}')
    plt.xlabel('Samples per Symbol (sps)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True)

    # -----------------------
    # 2) SER vs Pilot Length
    # -----------------------
    print('\n--- Plot 2: SER vs. Pilot Length ---')
    pilot_len_vec = [32, 64, 128, 256, 512]
    ser_vs_pilot = np.zeros(len(pilot_len_vec))
    N_const = 2048
    M_const = M_vec[1]
    sps_const = 8
    k_const = int(log2(M_const))

    for i, pilot_len in enumerate(pilot_len_vec):
        print(f"Testing pilot length = {pilot_len} ...")
        pilot_bits = np.random.randint(0, 2, pilot_len * k_const)
        data_bits = np.random.randint(0, 2, N_const * k_const)

        tx_pilot_syms = mod_func(pilot_bits, M_const)
        tx_data_syms = mod_func(data_bits, M_const)
        tx_syms = np.concatenate((tx_pilot_syms, tx_data_syms))

        pilot_upsampled = np.repeat(tx_pilot_syms, sps_const)
        tx_signal = np.repeat(tx_syms, sps_const)

        tx_buffer[:] = 0
        tx_buffer[:len(tx_signal)] = tx_signal
        sdr.start_tx(tx_buffer)
        time.sleep(0.3)
        received_data = sdr.rx()

        corr = np.abs(correlate(received_data, np.conj(pilot_upsampled), mode='valid'))
        peak_idx = int(np.argmax(corr)) if corr.size > 0 else 0
        start_idx = peak_idx
        rx_aligned = circ_extract(received_data, start_idx, len(tx_signal))
        rx_down = rx_aligned[::sps_const]

        rx_pilot_down = rx_down[:pilot_len]
        with np.errstate(divide='ignore', invalid='ignore'):
            h_est = np.mean(rx_pilot_down / tx_pilot_syms)
            if np.abs(h_est) == 0 or np.isnan(h_est):
                h_est = 1.0 + 0j
        rx_equalized = rx_down / h_est

        rx_bits = demod_func(rx_equalized, M_const)
        rx_data_bits = rx_bits[pilot_len * k_const:]

        tx_symbols_int = bits_to_integers(data_bits, k_const)
        rx_symbols_int = bits_to_integers(rx_data_bits, k_const)
        minlen = min(len(tx_symbols_int), len(rx_symbols_int))
        errs, ser = (0, 1.0) if minlen == 0 else symbol_error_rate(tx_symbols_int[:minlen], rx_symbols_int[:minlen])
        ser_vs_pilot[i] = ser
        print(f"  SER = {ser:.6f}")

    plt.figure()
    plt.semilogy(pilot_len_vec, ser_vs_pilot, 'ro-')
    plt.title(f'SER vs Pilot Length for {modulation_type}')
    plt.xlabel('Pilot Length (symbols)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True)

    # -----------------------
    # 3) SER vs Modulation Order (M)
    # -----------------------
    print('\n--- Plot 3: SER vs Modulation Order (M) ---')
    ser_vs_M = np.zeros(len(M_vec))
    N_const = 2048
    pilot_len_const = 128
    sps_const = 8

    for i, M in enumerate(M_vec):
        k = int(log2(M))
        print(f"Testing M = {M} ...")
        pilot_bits = np.random.randint(0, 2, pilot_len_const * k)
        data_bits = np.random.randint(0, 2, N_const * k)

        tx_pilot_syms = mod_func(pilot_bits, M)
        tx_data_syms = mod_func(data_bits, M)
        tx_syms = np.concatenate((tx_pilot_syms, tx_data_syms))

        pilot_upsampled = np.repeat(tx_pilot_syms, sps_const)
        tx_signal = np.repeat(tx_syms, sps_const)

        tx_buffer[:] = 0
        tx_buffer[:len(tx_signal)] = tx_signal
        sdr.start_tx(tx_buffer)
        time.sleep(0.3)
        received_data = sdr.rx()

        corr = np.abs(correlate(received_data, np.conj(pilot_upsampled), mode='valid'))
        peak_idx = int(np.argmax(corr)) if corr.size > 0 else 0
        start_idx = peak_idx
        rx_aligned = circ_extract(received_data, start_idx, len(tx_signal))
        rx_down = rx_aligned[::sps_const]

        rx_pilot_down = rx_down[:pilot_len_const]
        with np.errstate(divide='ignore', invalid='ignore'):
            h_est = np.mean(rx_pilot_down / tx_pilot_syms)
            if np.abs(h_est) == 0 or np.isnan(h_est):
                h_est = 1.0 + 0j
        rx_equalized = rx_down / h_est

        rx_bits = demod_func(rx_equalized, M)
        rx_data_bits = rx_bits[pilot_len_const * k: ]

        tx_symbols_int = bits_to_integers(data_bits, k)
        rx_symbols_int = bits_to_integers(rx_data_bits, k)
        minlen = min(len(tx_symbols_int), len(rx_symbols_int))
        errs, ser = (0, 1.0) if minlen == 0 else symbol_error_rate(tx_symbols_int[:minlen], rx_symbols_int[:minlen])
        ser_vs_M[i] = ser
        print(f"  SER = {ser:.6f}")

    plt.figure()
    plt.semilogy([int(m) for m in M_vec], ser_vs_M, 'go-')
    plt.title(f'SER vs Modulation Order (M) for {modulation_type}')
    plt.xlabel('Modulation Order (M)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True)

    # -----------------------
    # 4) SER vs N (data symbols)
    # -----------------------
    print('\n--- Plot 4: SER vs Number of Data Symbols (N) ---')
    N_vec = [512, 1024, 2048, 4096]
    ser_vs_N = np.zeros(len(N_vec))
    M_const = M_vec[1]
    pilot_len_const = 128
    sps_const = 8
    k_const = int(log2(M_const))

    for i, N in enumerate(N_vec):
        print(f"Testing N = {N} ...")
        pilot_bits = np.random.randint(0, 2, pilot_len_const * k_const)
        data_bits = np.random.randint(0, 2, N * k_const)

        tx_pilot_syms = mod_func(pilot_bits, M_const)
        tx_data_syms = mod_func(data_bits, M_const)
        tx_syms = np.concatenate((tx_pilot_syms, tx_data_syms))

        pilot_upsampled = np.repeat(tx_pilot_syms, sps_const)
        tx_signal = np.repeat(tx_syms, sps_const)

        tx_buffer[:] = 0
        tx_buffer[:len(tx_signal)] = tx_signal
        sdr.start_tx(tx_buffer)
        time.sleep(0.3)
        received_data = sdr.rx()

        corr = np.abs(correlate(received_data, np.conj(pilot_upsampled), mode='valid'))
        peak_idx = int(np.argmax(corr)) if corr.size > 0 else 0
        start_idx = peak_idx
        rx_aligned = circ_extract(received_data, start_idx, len(tx_signal))
        rx_down = rx_aligned[::sps_const]

        rx_pilot_down = rx_down[:pilot_len_const]
        with np.errstate(divide='ignore', invalid='ignore'):
            h_est = np.mean(rx_pilot_down / tx_pilot_syms)
            if np.abs(h_est) == 0 or np.isnan(h_est):
                h_est = 1.0 + 0j
        rx_equalized = rx_down / h_est

        rx_bits = demod_func(rx_equalized, M_const)
        rx_data_bits = rx_bits[pilot_len_const * k_const: ]

        tx_symbols_int = bits_to_integers(data_bits, k_const)
        rx_symbols_int = bits_to_integers(rx_data_bits, k_const)
        minlen = min(len(tx_symbols_int), len(rx_symbols_int))
        errs, ser = (0, 1.0) if minlen == 0 else symbol_error_rate(tx_symbols_int[:minlen], rx_symbols_int[:minlen])
        ser_vs_N[i] = ser
        print(f"  SER = {ser:.6f}")

    plt.figure()
    plt.semilogy(N_vec, ser_vs_N, 'mo-')
    plt.title(f'SER vs Number of Data Symbols for {modulation_type}')
    plt.xlabel('Number of Data Symbols (N)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True)

    # show all plots
    plt.show()

    print('--- All experiments finished ---')

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    # You can change modulation_type to 'PSK' if desired
    run_experiments(modulation_type='QAM')
