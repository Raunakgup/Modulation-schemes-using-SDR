import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
import time
import adi

# Pluto-only QAM version of the robust channel estimation demo
# -----------------------------------------------------------
# This file mirrors the Pluto-only PSK implementation but uses M-QAM
# modulation/demodulation everywhere. It transmits frames on a connected
# ADI PlutoSDR, receives the over-the-air capture, estimates the channel,
# equalizes and demodulates, and produces the same plots as the PSK version.

# --------------------------- User-configurable globals ----------------------
PLUTO_IP = "ip:192.168.2.1"  # Pluto device URI (change as needed)
PLUTO_CENTER_FREQ = 915e6    # RF center frequency in Hz

# Optional fixed RNG seed for reproducibility during debugging; remove to
# allow different frames each run.
np.random.seed(2)

# --------------------------- Utility helper functions ----------------------

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


def qammod(symbols, M=16):
    """Square M-QAM modulator (natural mapping, normalized to unit average power).

    symbols: integer symbols 0..M-1
    M must be a perfect square (e.g., 4, 16, 64)
    returns complex128 constellation points
    """
    syms = np.asarray(symbols).astype(int).ravel()
    # ensure M is perfect square
    m_side = int(round(sqrt(M)))
    if m_side * m_side != M:
        raise ValueError("M must be a perfect square for square QAM (e.g., 16,64)")

    # Generate grid points: coordinates are odd integers centered on zero.
    # For m_side = 4 -> coords = [-3, -1, 1, 3]
    coords = np.arange(-(m_side-1), m_side, 2)
    # produce meshgrid (real, imag) and flatten into constellation list
    re, im = np.meshgrid(coords, coords[::-1])  # reverse rows so mapping goes top->bottom
    const = (re + 1j*im).reshape(-1)

    # Natural mapping: symbol index k mapped to const[k]
    const = const.astype(np.complex128)

    # Normalize average power to 1
    power = np.mean(np.abs(const)**2)
    const = const / np.sqrt(power)

    # Map syms to constellation values (wrap modulo M)
    syms = np.mod(syms, M)
    return const[syms]


def qamdemod(symbols, M=16):
    """Minimum-distance M-QAM demodulator. Returns integer decisions 0..M-1."""
    symbols = np.asarray(symbols).ravel().astype(np.complex128)
    m_side = int(round(sqrt(M)))
    if m_side * m_side != M:
        raise ValueError("M must be a perfect square for square QAM")

    coords = np.arange(-(m_side-1), m_side, 2)
    re, im = np.meshgrid(coords, coords[::-1])
    const = (re + 1j*im).reshape(-1).astype(np.complex128)
    power = np.mean(np.abs(const)**2)
    const = const / np.sqrt(power)

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


# --------------------------- PlutoSDR wrapper -------------------------------
class PlutoSDRWrapper:
    def __init__(self, ip=PLUTO_IP, sample_rate=1e6, center_freq=PLUTO_CENTER_FREQ, rx_buffer_size=1024):
        self.sdr = adi.Pluto(ip)
        self.sample_rate = int(sample_rate)
        self.center_freq = int(center_freq)
        self.sdr.sample_rate = self.sample_rate

        self.sdr.tx_rf_bandwidth = self.sample_rate
        self.sdr.tx_lo = self.center_freq
        self.sdr.rx_lo = self.center_freq
        self.sdr.rx_rf_bandwidth = self.sample_rate

        self.sdr.rx_buffer_size = int(rx_buffer_size)

    def set_tx_gain(self, db):
        self.sdr.tx_hardwaregain_chan0 = float(db)

    def set_rx_gain(self, db, mode='manual'):
        self.sdr.gain_control_mode_chan0 = mode
        self.sdr.rx_hardwaregain_chan0 = float(db)

    def tx_waveform(self, samples, cyclic=True):
        self.sdr.tx_cyclic_buffer = bool(cyclic)
        self.sdr.tx(samples)

    def stop_tx(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass

    def rx_once(self):
        return self.sdr.rx()

    def rx_flush(self, n=5, delay_s=0.01):
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
    reps = ceil(sync_len_bits / len(barker))
    seq = np.tile(barker, reps)[:sync_len_bits]
    return (seq < 0).astype(int)


def run_single_frame_demo(Nbits=4000, M=16, sps=8, fs=1e6,
                          sync_barker13=None, ch_pilot_len_bits=128,
                          seed=None, pluto_ip=PLUTO_IP):
    if seed is not None:
        np.random.seed(seed)

    if sync_barker13 is None:
        sync_barker13 = DEFAULT_BARKER13
    sync_barker13 = np.tile(sync_barker13, 3)
    sync_len_bits = len(sync_barker13)

    print(f"INFO: [PARAM] Nbits={Nbits}, M={M}, sps={sps}, sync_len={sync_len_bits}, chpilot_len={ch_pilot_len_bits}")

    # Build frame
    sync_bits = (sync_barker13 < 0).astype(int)
    chpilot_bits = np.random.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = np.random.randint(0, M, size=(Nbits,))

    sync_symbols = qammod(sync_bits, M)
    chpilot_symbols = qammod(chpilot_bits, M)
    data_symbols = qammod(data_bits, M)

    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps
    print(f"INFO: [TX] frame_bits={frame_bits}, frame_samples={frame_samples}")
    print_snip('tx_frame (first)', tx_frame[:min(12, tx_frame.size)])

    # Pluto TX/RX
    print("INFO: [PLUTO] Initializing PlutoSDR and transmitting frame...")
    pluto = PlutoSDRWrapper(ip=pluto_ip, sample_rate=fs, center_freq=PLUTO_CENTER_FREQ, rx_buffer_size=frame_samples)
    pluto.set_tx_gain(-40)
    pluto.set_rx_gain(0.0)

    tx_samples = (tx_frame * (2**14)).astype(np.complex64)
    pluto.tx_waveform(tx_samples, cyclic=True)
    pluto.rx_flush(n=5)
    rx_samples = pluto.rx_once()
    pluto.stop_tx()
    pluto.close()

    rx_no_noise = rx_samples.astype(np.complex128)
    rx_received = rx_no_noise
    print_snip('rx_no_noise (first)', rx_no_noise[:min(12, rx_no_noise.size)])

    # Sync detection using doubled buffer
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

    # Channel pilot extraction
    chpilot_start_no = pilot_start_no + len(sync_shaped)
    chpilot_indices_no = (np.arange(chpilot_start_no, chpilot_start_no + ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_no = rx_no_noise[chpilot_indices_no]
    chpilot_mat_no = chpilot_samples_no.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_no = np.sum(chpilot_mat_no, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_no (first)', rx_chpilot_symbols_no[:min(12, rx_chpilot_symbols_no.size)])

    h_est_no = np.mean(rx_chpilot_symbols_no / chpilot_symbols)
    print(f"INFO: [H_EST_NO] h_est_no={h_est_no.real:.4f}{h_est_no.imag:+.4f}j")

    # Data extraction and demod
    data_start_no = chpilot_start_no + ch_pilot_len_bits * sps
    data_indices_no = (np.arange(data_start_no, data_start_no + Nbits*sps) % frame_samples).astype(int)
    data_samples_no = rx_no_noise[data_indices_no]
    data_mat_no = data_samples_no.reshape((sps, Nbits), order='F')
    symbol_samples_no = np.sum(data_mat_no, axis=0).astype(np.complex128)
    print_snip('symbol_samples_no (first)', symbol_samples_no[:min(12, symbol_samples_no.size)])

    rx_syms_before_no = qamdemod(symbol_samples_no, M)
    errs_before_no, ber_before_no = biterr(data_bits, rx_syms_before_no)
    print(f"INFO: [DEM_BEFORE_NO] NumErr={errs_before_no}, BER={ber_before_no:.6g}")

    symbol_samples_eq_no = symbol_samples_no / h_est_no
    print_snip('symbol_samples_eq_no (first)', symbol_samples_eq_no[:min(12, symbol_samples_eq_no.size)])

    rx_syms_after_no = qamdemod(symbol_samples_eq_no, M)
    errs_after_no, ber_after_no = biterr(data_bits, rx_syms_after_no)
    print(f"INFO: [DEM_AFTER_NO] NumErr={errs_after_no}, BER={ber_after_no:.6g}")

    # Received buffer branch (same variables retained for plotting)
    rx2_aw = np.concatenate([rx_received, rx_received])
    corr_aw = np.abs(np.convolve(rx2_aw, np.conjugate(sync_shaped[::-1])))
    peak_aw = np.argmax(corr_aw)
    pilot_start_in_rx2_aw = peak_aw - (len(sync_shaped) - 1)
    pilot_start_aw = pilot_start_in_rx2_aw % frame_samples
    print(f"INFO: [SYNC_AW] peak_aw={peak_aw}, pilot_start_aw={pilot_start_aw+1}, pilot_start mod sps={(pilot_start_aw % sps)+1}")

    seglen_aw = min(len(sync_shaped), frame_samples - pilot_start_aw)
    seg_aw = rx_received[pilot_start_aw: (pilot_start_aw + seglen_aw)]
    match_frac_aw = np.sum(np.abs(seg_aw - sync_shaped[:seglen_aw]) < 1e-6) / seglen_aw
    print(f"INFO: [SYNC_MATCH_AW] seglen={seglen_aw}, match_frac={match_frac_aw:.4f}")

    chpilot_start_aw = pilot_start_aw + len(sync_shaped)
    chpilot_indices_aw = (np.arange(chpilot_start_aw, chpilot_start_aw + ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples_aw = rx_received[chpilot_indices_aw]
    chpilot_mat_aw = chpilot_samples_aw.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_aw = np.sum(chpilot_mat_aw, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_aw (first)', rx_chpilot_symbols_aw[:min(12, rx_chpilot_symbols_aw.size)])

    h_est_aw = np.mean(rx_chpilot_symbols_aw / chpilot_symbols)
    print(f"INFO: [H_EST_AW] h_est_aw={h_est_aw.real:.4f}{h_est_aw.imag:+.4f}j")

    data_start_aw = chpilot_start_aw + ch_pilot_len_bits * sps
    data_indices_aw = (np.arange(data_start_aw, data_start_aw + Nbits*sps) % frame_samples).astype(int)
    data_samples_aw = rx_received[data_indices_aw]
    data_mat_aw = data_samples_aw.reshape((sps, Nbits), order='F')
    symbol_samples_aw = np.sum(data_mat_aw, axis=0).astype(np.complex128)
    print_snip('symbol_samples_aw (first)', symbol_samples_aw[:min(12, symbol_samples_aw.size)])

    symbol_samples_eq_aw = symbol_samples_aw / h_est_aw
    print_snip('symbol_samples_eq_aw (first)', symbol_samples_eq_aw[:min(12, symbol_samples_eq_aw.size)])

    # ----------------------- Debug: Tx vs Rx (first 20 symbols) -----------------------
    n_plot = 20  # number of symbols to compare
    n_plot = min(n_plot, data_symbols.size, symbol_samples_eq_aw.size)
    tx_plot = data_symbols[:n_plot]
    rx_plot = symbol_samples_eq_aw[:n_plot]
    plt.figure('Tx vs Rx Constellation (first symbols)', figsize=(6, 6))
    plt.plot(np.real(tx_plot), np.imag(tx_plot), 'x', markersize=8, label='Transmitted (first {})'.format(n_plot))
    plt.plot(np.real(rx_plot), np.imag(rx_plot), 'o', markersize=6, label='Received EQ (first {})'.format(n_plot))
    for i in range(n_plot):
        t = tx_plot[i]
        r = rx_plot[i]
        plt.plot([np.real(t), np.real(r)], [np.imag(t), np.imag(r)], '-', linewidth=0.7, alpha=0.6)
    plt.title('Transmitted (x) vs Received after EQ (o) — first {} symbols'.format(n_plot))
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='best')
    inds = np.arange(n_plot)
    plt.figure('Tx vs Rx time series (first symbols)', figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(inds, np.real(tx_plot), 'x-', label='Tx real')
    plt.plot(inds, np.real(rx_plot), 'o-', label='Rx eq real')
    plt.ylabel('Real')
    plt.grid(True)
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(inds, np.imag(tx_plot), 'x-', label='Tx imag')
    plt.plot(inds, np.imag(rx_plot), 'o-', label='Rx eq imag')
    plt.xlabel('Symbol index (0 = first data symbol)')
    plt.ylabel('Imag')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    rx_syms_before_aw = qamdemod(symbol_samples_aw, M)
    errs_before_aw, ber_before_aw = biterr(data_bits, rx_syms_before_aw)
    print(f"INFO: [DEM_BEFORE_AW] NumErr={errs_before_aw}, BER={ber_before_aw:.6g}")

    rx_syms_after_aw = qamdemod(symbol_samples_eq_aw, M)
    errs_after_aw, ber_after_aw = biterr(data_bits, rx_syms_after_aw)
    print(f"INFO: [DEM_AFTER_AW] NumErr={errs_after_aw}, BER={ber_after_aw:.6g}")

    plt.figure('No received: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_no[:2000]), np.imag(symbol_samples_no[:2000]), '.', markersize=6)
    plt.title('No received — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_no[:2000]), np.imag(symbol_samples_eq_no[:2000]), '.', markersize=6)
    plt.title('No received — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.figure('Received: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_aw[:2000]), np.imag(symbol_samples_aw[:2000]), '.')
    plt.title('Received buffer — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_aw[:2000]), np.imag(symbol_samples_eq_aw[:2000]), '.')
    plt.title('Received buffer — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.figure('Correlation peaks', figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(corr_no)
    plt.title('Correlation magnitude (no received)')
    plt.ylabel('mag')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(corr_aw)
    plt.title('Correlation magnitude (received buffer)')
    plt.ylabel('mag')
    plt.grid(True)

    plt.figure('Zoom After Equalization (received buffer)', figsize=(6, 6))
    plt.plot(np.real(symbol_samples_eq_aw), np.imag(symbol_samples_eq_aw), '.')
    plt.title('After equalization (received buffer) - zoom subset')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.axis('equal')

    print(f"\nINFO: [SUMMARY] NO_received: BER_before={ber_before_no:.6g} BER_after={ber_after_no:.6g} | RECEIVED: BER_before={ber_before_aw:.6g} BER_after={ber_after_aw:.6g}")

    return {
        'ber_noreceived_before': ber_before_no,
        'ber_noreceived_after': ber_after_no,
        'ber_received_before': ber_before_aw,
        'ber_received_after': ber_after_aw,
        'symbol_samples_eq_aw': symbol_samples_eq_aw,
        'symbol_samples_eq_no': symbol_samples_eq_no,
        'corr_no': corr_no,
        'corr_aw': corr_aw,
    }


# ------------------- Monte Carlo helpers (Pluto-only) ---------------------
def simulate_frame_ser(M=16, sps=8, fs=1e6, Nbits=4000,
                       sync_len_bits=26, ch_pilot_len_bits=128,
                       rng=None, pluto_ip=PLUTO_IP):
    if rng is None:
        rng = np.random

    sync_bits = _make_sync_bits(sync_len_bits)
    chpilot_bits = rng.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = rng.randint(0, M, size=(Nbits,))

    sync_symbols = qammod(sync_bits, M)
    chpilot_symbols = qammod(chpilot_bits, M)
    data_symbols = qammod(data_bits, M)

    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps

    pluto = PlutoSDRWrapper(ip=pluto_ip, sample_rate=fs, center_freq=PLUTO_CENTER_FREQ, rx_buffer_size=frame_samples)
    pluto.set_tx_gain(-40)
    pluto.set_rx_gain(0.0)

    tx_samples = (tx_frame * (2**14)).astype(np.complex64)
    pluto.tx_waveform(tx_samples, cyclic=True)
    pluto.rx_flush(n=5)
    rx = pluto.rx_once().astype(np.complex128)
    pluto.stop_tx()
    pluto.close()

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

    data_start = chpilot_start + ch_pilot_len_bits*sps
    data_indices = (np.arange(data_start, data_start + Nbits*sps) % frame_samples).astype(int)
    data_samples = rx[data_indices]
    data_mat = data_samples.reshape((sps, Nbits), order='F')
    symbol_samples = np.sum(data_mat, axis=0).astype(np.complex128)

    symbol_samples_eq = symbol_samples / h_est
    rx_syms = qamdemod(symbol_samples_eq, M)
    num_errors = np.sum(rx_syms != data_bits)
    return int(num_errors), int(Nbits)


def monte_carlo_ser(n_trials=100, seed=None, **simulate_kwargs):
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


# ------------------------ Plot helpers (QAM) -------------------------------
def plot_ser_vs_sps(fs, M, Nbits, sps_list, ch_pilot_len_bits=128,
                    sync_len_bits=26, n_trials=80):
    sers = []
    for sps in sps_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits)
        print(f"INFO: sps={sps} SER={ser:.4e}")
        sers.append(ser)
    sps_arr = np.array(sps_list)
    Rs_arr = fs / sps_arr
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(sps_arr, sers, '-o')
    plt.title(f'SER vs sps (fs={fs}, M={M}, N={Nbits})')
    plt.xlabel('samples per symbol (sps)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.semilogy(Rs_arr, sers, '-o')
    plt.title('SER vs Symbol Rate Rs')
    plt.xlabel('Rs = fs / sps (symbols/sec)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return sps_arr, Rs_arr, np.array(sers)


def plot_ser_vs_sync_len(fs, M, Nbits, sync_len_list, sps=8,
                         ch_pilot_len_bits=128, n_trials=80):
    sers = []
    for sync_len in sync_len_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len)
        print(f"INFO: sync_len={sync_len} SER={ser:.4e}")
        sers.append(ser)
    plt.figure(figsize=(7, 5))
    plt.semilogy(sync_len_list, sers, '-o')
    plt.title(f'SER vs sync length (fs={fs}, M={M}, sps={sps}, N={Nbits})')
    plt.xlabel('sync / pilot length (symbols)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(sync_len_list), np.array(sers)


def plot_ser_vs_M(fs, M_list, Nbits, sps=8, ch_pilot_len_bits=128,
                  sync_len_bits=26, n_trials=80):
    sers = []
    for M in M_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits)
        print(f"INFO: M={M} SER={ser:.4e}")
        sers.append(ser)
    plt.figure(figsize=(7, 5))
    plt.semilogy(M_list, sers, '-o')
    plt.title(f'SER vs Modulation order M (fs={fs}, N={Nbits}, sps={sps})')
    plt.xlabel('M (QAM order)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(M_list), np.array(sers)


def plot_ser_vs_N(fs, M, sps, N_list, ch_pilot_len_bits=128, sync_len_bits=26,
                  n_trials=80):
    sers = []
    for Nbits in N_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits)
        print(f"INFO: N={Nbits} SER={ser:.4e}")
        sers.append(ser)
    plt.figure(figsize=(7, 5))
    plt.semilogy(N_list, sers, '-o')
    plt.title(f'SER vs Number of data symbols N (fs={fs}, M={M}, sps={sps})')
    plt.xlabel('Number of data symbols (N)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(N_list), np.array(sers)


def find_best_params_for_M(M_list, sps_candidates, N_candidates, fs_candidates,
                           ch_pilot_len_bits=128, sync_len_bits=26,
                           n_trials=60, top_k=5):
    results = {}
    for M in M_list:
        combos = []
        for sps in sps_candidates:
            for N in N_candidates:
                for fs in fs_candidates:
                    ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                                M=M, sps=sps, fs=fs, Nbits=N,
                                                ch_pilot_len_bits=ch_pilot_len_bits,
                                                sync_len_bits=sync_len_bits)
                    Rs = fs / sps
                    combos.append((ser, sps, N, fs, Rs))
                    print(f"INFO: M={M} sps={sps} N={N} fs={fs:.2e} Rs={Rs:.2f} SER={ser:.4e}")
        combos.sort(key=lambda x: x[0])
        results[M] = combos[:top_k]
    return results


# ------------------------------ Main guard --------------------------------
if __name__ == "__main__":
    # 1) Run single-frame demo and show its constellation/correlation plots
    demo_results = run_single_frame_demo(Nbits=2000, M=16, sps=8, fs=1e6, seed=2)

    # 2) Example sweeps for the five objectives (note: each monte_carlo trial
    # transmits/receives a frame on the Pluto, so keep n_trials modest for speed).
    fs = 1e6
    M = 16
    Nbits = 2000

    # Objective (i): SER vs sps
    sps_list = [1, 2, 4, 8, 16]
    plot_ser_vs_sps(fs=fs, M=M, Nbits=Nbits, sps_list=sps_list, n_trials=60)

    # Objective (ii): SER vs sync length
    sync_len_list = [8, 13, 26, 52]
    plot_ser_vs_sync_len(fs=fs, M=M, Nbits=Nbits, sync_len_list=sync_len_list, sps=8, n_trials=60)

    # Objective (iii): SER vs modulation order M
    plot_ser_vs_M(fs=fs, M_list=[4, 16, 64], Nbits=1000, sps=8, n_trials=60)

    # Objective (iv): SER vs number of data symbols N
    plot_ser_vs_N(fs=fs, M=M, sps=8, N_list=[500, 1000, 2000, 4000], n_trials=60)

    # Objective (v): coarse search for best params
    results = find_best_params_for_M(M_list=[4, 16], sps_candidates=[2, 4, 8], N_candidates=[1000, 2000], fs_candidates=[1e6, 2e6], n_trials=40)
    print("\nBest combos (per M):")
    for Mkey, combos in results.items():
        print(f"M={Mkey}")
        for ser, sps_v, N_v, fs_v, Rs_v in combos:
            print(f"  SER={ser:.4e} sps={sps_v} N={N_v} fs={fs_v:.2e} Rs={Rs_v:.2f}")

    plt.show()
