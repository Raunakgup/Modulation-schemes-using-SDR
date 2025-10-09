"""
robust_channel_est_with_pluto.py

This is your original robust_channel_est_with_objectives.py updated with
optional ADALM-PLUTO (pyadi-iio) integration for real transmit/receive.

Key changes:
 - Added init_pluto, pluto_tx_rx helper functions using pyadi-iio (adi).
 - run_single_frame_demo gains a use_sdr flag (default False). When
   use_sdr=True the script will initialize the Pluto, transmit the frame
   and receive the same-length frame from the radio. In SDR mode AWGN and
   the simulated circular shift are NOT applied (they are done by the
   real channel / SDR hardware).
 - The simulation helpers are left intact and still usable when
   use_sdr=False.

NOTE: This requires pyadi-iio (pip install pyadi-iio) and a connected
ADALM-PLUTO. Default Pluto URI used is 'ip:192.168.2.1' (change via
sdr_uri).

"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import time
import adi
np.random.seed(1)

# Optional SDR import
try:
    import adi
    HAS_ADI = True
    print("adi imported")
except Exception:
    HAS_ADI = False

print(HAS_ADI)
# ---------------- Helpers (kept from your original code) ----------------


def print_snip(name, x):
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
    """
    General M-PSK modulator.
    Input: integer symbols in 0..M-1 (array-like)
    Output: complex constellation points exp(1j*2*pi*k/M)
    """
    syms = np.asarray(symbols).astype(int).ravel()
    syms = np.mod(syms, M)
    return np.exp(1j * 2.0 * np.pi * syms / M).astype(np.complex128)


def pskdemod(symbols, M=2):
    """
    General M-PSK demodulator by minimum Euclidean distance.
    Returns integer symbols in 0..M-1.
    """
    symbols = np.asarray(symbols).ravel().astype(np.complex128)
    const = np.exp(1j * 2.0 * np.pi * np.arange(M) / M).astype(np.complex128)
    diff = symbols.reshape(-1, 1) - const.reshape(1, -1)
    dist = np.abs(diff)
    decisions = np.argmin(dist, axis=1).astype(int)
    return decisions


def biterr(a, b):
    a = np.asarray(a).ravel().astype(int)
    b = np.asarray(b).ravel().astype(int)
    errs = np.sum(a != b)
    ber = errs / a.size
    return int(errs), float(ber)

def awgn(sig, snr_db):
    sig = np.asarray(sig)
    power_signal = np.mean(np.abs(sig)**2)
    snr_linear = 10**(snr_db/10.0)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*
                                                      sig.shape) + 1j*np.random.randn(*sig.shape))
    return sig + noise


# ---------------- Pluto SDR helpers ----------------


def init_pluto(sample_rate=1e6, center_freq=915e6, tx_gain=-10.0, rx_gain=30.0,
               uri: str = "ip:192.168.2.10", rx_buffer_size=None, tx_cyclic_buffer=True):
    """
    Initialize ADALM-PLUTO via pyadi-iio (adi.Pluto).

    Returns the adi.Pluto object. Raises ImportError if pyadi-iio is not
    installed or RuntimeError if the device cannot be opened.

    Common attributes used (pyadi-iio):
      - sample_rate
      - tx_lo, rx_lo
      - tx_rf_bandwidth, rx_rf_bandwidth
      - tx_hardwaregain_chan0 (TX attenuation in dB)
      - gain_control_mode_chan0 and rx_hardwaregain_chan0 (RX gain in dB)
      - tx_cyclic_buffer (True/False)
      - rx_buffer_size

    Notes: call with the physical/pluto URI (eg 'ip:192.168.2.1' or 'usb:0').
    """
    if not HAS_ADI:
        raise ImportError("pyadi-iio (adi) package not found. Install with pip install pyadi-iio")

    print(f"INFO: Initializing Pluto SDR at '{uri}' (fs={sample_rate}, fc={center_freq})")
    sdr = adi.Pluto()

    # Basic RF/sample-rate settings
    sdr.sample_rate = int(sample_rate)
    sdr.tx_lo = int(center_freq)
    sdr.rx_lo = int(center_freq)
    # Set RF filter bandwidths to sample rate as a reasonable default
    try:
        sdr.tx_rf_bandwidth = int(sample_rate)
        sdr.rx_rf_bandwidth = int(sample_rate)
    except Exception:
        # Not all firmwares/hardware accept these; ignore if so
        pass

    # Gains: tx_hardwaregain_chan0 is the TX attenuation value (dB)
    try:
        sdr.tx_hardwaregain_chan0 = float(tx_gain)
    except Exception:
        # attribute name might differ by version: try generic
        try:
            sdr.tx_hardwaregain = float(tx_gain)
        except Exception:
            print("WARNING: Could not set TX gain attribute on Pluto (continuing)")

    # RX: set manual gain mode and the RX gain
    try:
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0 = float(rx_gain)
    except Exception:
        try:
            sdr.gain_control_mode = 'manual'
            sdr.rx_hardwaregain = float(rx_gain)
        except Exception:
            print("WARNING: Could not set RX gain attribute on Pluto (continuing)")

    # Buffering
    sdr.tx_cyclic_buffer = bool(tx_cyclic_buffer)
    if rx_buffer_size is not None:
        sdr.rx_buffer_size = int(rx_buffer_size)

    # small settle
    time.sleep(0.05)
    print("INFO: Pluto initialized")
    return sdr


def pluto_tx_rx(sdr, tx_samples: np.ndarray, rx_len: int, tx_scale: float = 0.6, rx_timeout_s: float = 5.0):
    """
    Transmit tx_samples (1D complex array) and receive rx_len complex
    samples back from the device. Returns the received complex numpy array.

    - tx_samples should be dtype=np.complex64 (or will be cast).
    - The function will enable cyclic transmit when sdr.tx_cyclic_buffer is
      True so the transmitter keeps sending while we call rx().

    Note: hardware behavior (timing / underflow) depends on platform/firmware.
    """
    if not HAS_ADI:
        raise ImportError("pyadi-iio (adi) package not found. Install with pip install pyadi-iio")

    # Cast and scale TX samples conservatively to avoid saturating the DAC.
    tx_arr = np.asarray(tx_samples).astype(np.complex64) * np.float32(tx_scale)

    # Start transmit. If cyclic buffer is True this will repeat until destroyed.
    print(f"INFO: Starting TX of {tx_arr.size} samples (cyclic={sdr.tx_cyclic_buffer})")
    sdr.tx(tx_arr)

    # Small sleep to allow TX to ramp and the hardware to settle
    time.sleep(0.01)

    # Read rx_len samples (may block until buffer fills)
    print(f"INFO: Receiving {rx_len} samples from Pluto (this may block until buffer is ready)")
    rx = sdr.rx(rx_len)

    # Try best-effort to stop cyclic TX if possible to release resources.
    try:
        if hasattr(sdr, 'tx_destroy_buffer'):
            sdr.tx_destroy_buffer()
        else:
            # toggle cyclic flag off (may not stop immediately depending on driver)
            sdr.tx_cyclic_buffer = False
            # push a short zero buffer to clear TX path if supported
            try:
                sdr.tx(np.zeros(4, dtype=np.complex64))
            except Exception:
                pass
    except Exception:
        # not critical, warn and continue
        print("WARNING: Could not explicitly destroy/stop TX buffer (driver may keep repeating until object destroyed)")

    # Return as complex128 for downstream processing (matches the rest of script)
    return np.asarray(rx).astype(np.complex128)


# ---------------- Single-frame demo (keeps all your prints + plots) ----------------
DEFAULT_BARKER13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=int)


def _make_sync_bits(sync_len_bits, barker=DEFAULT_BARKER13):
    reps = ceil(sync_len_bits / len(barker))
    seq = np.tile(barker, reps)[:sync_len_bits]
    return (seq < 0).astype(int)


def run_single_frame_demo(Nbits=4000, M=16, sps=8, fs=1e6,
                          snr_db=10, sync_barker13=None, ch_pilot_len_bits=128,
                          with_awgn=True, seed=None,
                          use_sdr=False, sdr_uri: str = "ip:192.168.2.1",
                          sdr_center_freq: float = 2.4e9,
                          sdr_tx_gain: float = -10.0,
                          sdr_rx_gain: float = 30.0,
                          sdr_tx_scale: float = 0.6,
                          sdr_rx_timeout_s: float = 5.0):
    """
    Run the single-frame demo. If use_sdr==False this behaves like your
    original simulator (it still uses AWGN and circular-shift to emulate an
    SDR). If use_sdr==True the function will initialize a Pluto SDR (via
    pyadi-iio), transmit the frame and receive a frame of the same length.

    When using the hardware we DO NOT apply AWGN or the simulated circular
    shift since the channel and timing will be produced by the real link.
    """
    if seed is not None:
        np.random.seed(seed)

    if sync_barker13 is None:
        sync_barker13 = DEFAULT_BARKER13
    sync_barker13 = np.tile(sync_barker13, 3)  # default robust repetition
    sync_len_bits = len(sync_barker13)

    print(
        f"INFO: [PARAM] Nbits={Nbits}, M={M}, sps={sps}, sync_len={sync_len_bits}, chpilot_len={ch_pilot_len_bits}, snr_db={snr_db}")

    # Build transmit frame (symbols)
    sync_bits = (sync_barker13 < 0).astype(int)
    chpilot_bits = np.random.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = np.random.randint(0, M, size=(Nbits,))

    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    print_snip('sync_bits', sync_bits)
    print_snip('sync_symbols', sync_symbols)
    print_snip('chpilot_bits', chpilot_bits)
    print_snip('chpilot_symbols', chpilot_symbols)
    print_snip('data_bits (first)', data_bits[:min(12, data_bits.size)])
    print_snip('data_symbols (first)',
               data_symbols[:min(12, data_symbols.size)])

    # NRZ shaping (repeat ensures exact sps alignment)
    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    print(
        f"INFO: [SHAPE] sizes: sync_shaped={len(sync_shaped)}, chpilot_shaped={len(chpilot_shaped)}, data_shaped={len(data_shaped)}")
    print_snip('sync_shaped (first samples)',
               sync_shaped[:min(12, sync_shaped.size)])

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps
    print(f"INFO: [TX] frame_bits={frame_bits}, frame_samples={frame_samples}")
    print_snip('tx_frame (first)', tx_frame[:min(12, tx_frame.size)])

    # If using SDR: transmit via Pluto and receive a frame of the same length.
    if use_sdr:
        if not HAS_ADI:
            raise ImportError("pyadi-iio (adi) not available; cannot use Pluto SDR mode")

        sdr = init_pluto(sample_rate=fs, center_freq=sdr_center_freq,
                         tx_gain=sdr_tx_gain, rx_gain=sdr_rx_gain,
                         uri=sdr_uri, rx_buffer_size=frame_samples,
                         tx_cyclic_buffer=True)

        # Convert to complex baseband float and send/receive
        tx_iq = np.asarray(tx_frame).astype(np.complex64)
        rx_iq = pluto_tx_rx(sdr, tx_iq, rx_len=frame_samples,
                            tx_scale=sdr_tx_scale, rx_timeout_s=sdr_rx_timeout_s)

        # Use the device-provided samples as 'rx_no_noise'
        rx_no_noise = rx_iq.copy()
        rx_awgn = rx_no_noise.copy()  # no extra AWGN when using real SDR

        # NOTE: do not destroy sdr here; we leave it to user or garbage collector
    else:
        # SOFTWARE SIMULATION PATH (unchanged)
        attenuation = (np.random.rand() + 1j*np.random.rand())
        print(
            f"INFO: [CH] True attenuation = {attenuation.real:.4f}{attenuation.imag:+.4f}j")
        attenuated_tx = attenuation * tx_frame

        # SDR-like circular shift
        random_index = np.random.randint(1, frame_samples+1)
        idx0 = random_index - 1
        rx_no_noise = np.concatenate([attenuated_tx[idx0:], attenuated_tx[:idx0]])
        print(f"INFO: [RX_SHIFT] random_index={random_index}")
        print_snip('rx_no_noise (first)', rx_no_noise[:min(12, rx_no_noise.size)])

        # AWGN version
        rx_awgn = awgn(rx_no_noise, snr_db) if with_awgn else rx_no_noise.copy()
        print(f"INFO: [AWGN] Applied AWGN at {snr_db} dB (with_awgn={with_awgn})")
        print_snip('rx_awgn (first)', rx_awgn[:min(12, rx_awgn.size)])

    # From here on we reuse your existing sync/pilot/data extraction logic.

    # Correlation helper & sync detection (no AWGN)
    rx2_no = np.concatenate([rx_no_noise, rx_no_noise])
    corr_no = np.abs(np.convolve(rx2_no, np.conjugate(sync_shaped[::-1])))
    peak_no = np.argmax(corr_no)
    pilot_start_in_rx2_no = peak_no - (len(sync_shaped) - 1)
    pilot_start_no = pilot_start_in_rx2_no % frame_samples
    print(
        f"INFO: [SYNC_NO] peak_no={peak_no}, pilot_start_no={pilot_start_no+1}, pilot_start mod sps={(pilot_start_no % sps)+1}")

    seglen_no = min(len(sync_shaped), frame_samples - pilot_start_no)
    seg_no = rx_no_noise[pilot_start_no: (pilot_start_no + seglen_no)]
    match_frac_no = np.sum(
        np.abs(seg_no - sync_shaped[:seglen_no]) < 1e-6) / seglen_no
    print(
        f"INFO: [SYNC_MATCH_NO] seglen={seglen_no}, match_frac={match_frac_no:.4f}")

    # Channel pilot extraction (no AWGN)
    chpilot_start_no = pilot_start_no + len(sync_shaped)
    chpilot_indices_no = (np.arange(chpilot_start_no, chpilot_start_no +
                          ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_no = rx_no_noise[chpilot_indices_no]
    chpilot_mat_no = chpilot_samples_no.reshape(
        (sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_no = np.sum(
        chpilot_mat_no, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_no (first)',
               rx_chpilot_symbols_no[:min(12, rx_chpilot_symbols_no.size)])

    # Estimate channel (no AWGN)
    h_est_no = np.mean(rx_chpilot_symbols_no / chpilot_symbols)
    print(
        f"INFO: [H_EST_NO] h_est_no={h_est_no.real:.4f}{h_est_no.imag:+.4f}j")

    # Extract data symbols raw (before equalization)
    data_start_no = chpilot_start_no + ch_pilot_len_bits * sps
    data_indices_no = (np.arange(data_start_no, data_start_no +
                       Nbits*sps) % frame_samples).astype(int)
    data_samples_no = rx_no_noise[data_indices_no]
    data_mat_no = data_samples_no.reshape((sps, Nbits), order='F')
    symbol_samples_no = np.sum(data_mat_no, axis=0).astype(np.complex128)
    print_snip('symbol_samples_no (first)',
               symbol_samples_no[:min(12, symbol_samples_no.size)])

    # Demod BEFORE equalization (no AWGN)
    rx_syms_before_no = pskdemod(symbol_samples_no, M)
    errs_before_no, ber_before_no = biterr(data_bits, rx_syms_before_no)
    print(
        f"INFO: [DEM_BEFORE_NO] NumErr={errs_before_no}, BER={ber_before_no:.6g}")

    # Equalize (no AWGN)
    symbol_samples_eq_no = symbol_samples_no / h_est_no
    print_snip('symbol_samples_eq_no (first)',
               symbol_samples_eq_no[:min(12, symbol_samples_eq_no.size)])

    rx_syms_after_no = pskdemod(symbol_samples_eq_no, M)
    errs_after_no, ber_after_no = biterr(data_bits, rx_syms_after_no)
    print(
        f"INFO: [DEM_AFTER_NO] NumErr={errs_after_no}, BER={ber_after_no:.6g}")

    # ---------- With AWGN ---------- (perform sync & pilot extraction on noisy rx)
    rx2_aw = np.concatenate([rx_awgn, rx_awgn])
    corr_aw = np.abs(np.convolve(rx2_aw, np.conjugate(sync_shaped[::-1])))
    peak_aw = np.argmax(corr_aw)
    pilot_start_in_rx2_aw = peak_aw - (len(sync_shaped) - 1)
    pilot_start_aw = pilot_start_in_rx2_aw % frame_samples
    print(
        f"INFO: [SYNC_AW] peak_aw={peak_aw}, pilot_start_aw={pilot_start_aw+1}, pilot_start mod sps={(pilot_start_aw % sps)+1}")

    seglen_aw = min(len(sync_shaped), frame_samples - pilot_start_aw)
    seg_aw = rx_awgn[pilot_start_aw: (pilot_start_aw + seglen_aw)]
    match_frac_aw = np.sum(
        np.abs(seg_aw - sync_shaped[:seglen_aw]) < 1e-6) / seglen_aw
    print(
        f"INFO: [SYNC_MATCH_AW] seglen={seglen_aw}, match_frac={match_frac_aw:.4f}")

    # channel pilot under AWGN
    chpilot_start_aw = pilot_start_aw + len(sync_shaped)
    chpilot_indices_aw = (np.arange(chpilot_start_aw, chpilot_start_aw +
                          ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_aw = rx_awgn[chpilot_indices_aw]
    chpilot_mat_aw = chpilot_samples_aw.reshape(
        (sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_aw = np.sum(
        chpilot_mat_aw, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_aw (first)',
               rx_chpilot_symbols_aw[:min(12, rx_chpilot_symbols_aw.size)])

    h_est_aw = np.mean(rx_chpilot_symbols_aw / chpilot_symbols)
    print(
        f"INFO: [H_EST_AW] h_est_aw={h_est_aw.real:.4f}{h_est_aw.imag:+.4f}j")

    # Extract data (noisy)
    data_start_aw = chpilot_start_aw + ch_pilot_len_bits * sps
    data_indices_aw = (np.arange(data_start_aw, data_start_aw +
                       Nbits*sps) % frame_samples).astype(int)
    data_samples_aw = rx_awgn[data_indices_aw]
    data_mat_aw = data_samples_aw.reshape((sps, Nbits), order='F')
    symbol_samples_aw = np.sum(data_mat_aw, axis=0).astype(np.complex128)
    print_snip('symbol_samples_aw (first)',
               symbol_samples_aw[:min(12, symbol_samples_aw.size)])

    # BEFORE equalization demod (AWGN)
    rx_syms_before_aw = pskdemod(symbol_samples_aw, M)
    errs_before_aw, ber_before_aw = biterr(data_bits, rx_syms_before_aw)
    print(
        f"INFO: [DEM_BEFORE_AW] NumErr={errs_before_aw}, BER={ber_before_aw:.6g}")

    # Equalize & demod (AWGN)
    symbol_samples_eq_aw = symbol_samples_aw / h_est_aw
    print_snip('symbol_samples_eq_aw (first)',
               symbol_samples_eq_aw[:min(12, symbol_samples_eq_aw.size)])
    rx_syms_after_aw = pskdemod(symbol_samples_eq_aw, M)
    errs_after_aw, ber_after_aw = biterr(data_bits, rx_syms_after_aw)
    print(
        f"INFO: [DEM_AFTER_AW] NumErr={errs_after_aw}, BER={ber_after_aw:.6g}")

    # ---------- Plots (constellations, correlations) ----------
    plt.figure('No AWGN: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_no[:2000]), np.imag(
        symbol_samples_no[:2000]), '.', markersize=6)
    plt.title('No AWGN — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_no[:2000]), np.imag(
        symbol_samples_eq_no[:2000]), '.', markersize=6)
    plt.title('No AWGN — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.figure('With AWGN: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_aw[:2000]),
             np.imag(symbol_samples_aw[:2000]), '.')
    plt.title(f'AWGN {snr_db} dB — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_aw[:2000]), np.imag(
        symbol_samples_eq_aw[:2000]), '.')
    plt.title(f'AWGN {snr_db} dB — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    # Correlation peaks
    plt.figure('Correlation peaks', figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(corr_no)
    plt.title('Correlation magnitude (no AWGN)')
    plt.ylabel('mag')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(corr_aw)
    plt.title('Correlation magnitude (with AWGN)')
    plt.ylabel('mag')
    plt.grid(True)

    plt.figure('Zoom After Equalization (AWGN)', figsize=(6, 6))
    plt.plot(np.real(symbol_samples_eq_aw), np.imag(symbol_samples_eq_aw), '.')
    plt.title('After equalization (AWGN) - zoom subset')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.axis('equal')

    print(
        f"\nINFO: [SUMMARY] NO_AWGN: BER_before={ber_before_no:.6g} BER_after={ber_after_no:.6g} | WITH_AWGN: BER_before={ber_before_aw:.6g} BER_after={ber_after_aw:.6g}")

    # Return a dict useful for external plotting / tests
    return {
        'ber_noawgn_before': ber_before_no,
        'ber_noawgn_after': ber_after_no,
        'ber_awgn_before': ber_before_aw,
        'ber_awgn_after': ber_after_aw,
        'symbol_samples_eq_aw': symbol_samples_eq_aw,
        'symbol_samples_eq_no': symbol_samples_eq_no,
        'corr_no': corr_no,
        'corr_aw': corr_aw,
    }

# ---------------- The rest of the simulation utilities / plotting helpers remain unchanged ----------------
# (For brevity they are not duplicated here - keep the original simulate_frame_ser,
# monte_carlo_ser and plotting helpers from your file if you need them.)

if __name__ == "__main__":
    # Quick interactive demo: set use_sdr=True to actually use a connected Pluto
    demo_results = run_single_frame_demo(
        Nbits=2000, M=16, sps=8, fs=1e6, snr_db=10, with_awgn=True, seed=1,
        use_sdr=True  # <-- set to True to use real Pluto hardware
    )

    plt.show()