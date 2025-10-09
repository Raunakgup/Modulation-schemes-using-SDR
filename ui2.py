#!/usr/bin/env python3
"""
robust_channel_est_with_objectives.py

Complete script: original single-frame demo + reusable simulation functions +
plotting helpers for the five objectives:
  i)  SER vs sps (and Rs)
  ii) SER vs sync/pilot length
  iii) SER vs modulation order M
  iv) SER vs number of data symbols N
  v)  coarse search for best (Rs, N, fs) combos per M

Warning: Monte Carlo sweeps may be compute heavy. Default trial counts are moderate.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

np.random.seed(1)

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


def awgn(sig, snr_db):
    sig = np.asarray(sig)
    power_signal = np.mean(np.abs(sig)**2)
    snr_linear = 10**(snr_db/10.0)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*
                                                      sig.shape) + 1j*np.random.randn(*sig.shape))
    return sig + noise


def biterr(a, b):
    a = np.asarray(a).ravel().astype(int)
    b = np.asarray(b).ravel().astype(int)
    errs = np.sum(a != b)
    ber = errs / a.size
    return int(errs), float(ber)


# ---------------- Single-frame demo (keeps all your prints + plots) ----------------
DEFAULT_BARKER13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=int)


def _make_sync_bits(sync_len_bits, barker=DEFAULT_BARKER13):
    reps = ceil(sync_len_bits / len(barker))
    seq = np.tile(barker, reps)[:sync_len_bits]
    return (seq < 0).astype(int)


def run_single_frame_demo(Nbits=4000, M=16, sps=8, fs=1e6,
                          snr_db=10, sync_barker13=None, ch_pilot_len_bits=128,
                          with_awgn=True, seed=None):
    """
    Run the original single-frame demo (random channel, circular shift) and
    show the constellation / correlation plots as in your original MATLAB demo.
    Returns a dict with key results and arrays for plotting if desired.
    """
    if seed is not None:
        np.random.seed(seed)

    if sync_barker13 is None:
        sync_barker13 = DEFAULT_BARKER13
    sync_barker13 = np.tile(sync_barker13, 3)  # default robust repetition
    sync_len_bits = len(sync_barker13)

    print(
        f"INFO: [PARAM] Nbits={Nbits}, M={M}, sps={sps}, sync_len={sync_len_bits}, chpilot_len={ch_pilot_len_bits}, snr_db={snr_db}")

    # Build transmit frame
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

    # Channel model (random complex scalar)
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
        f"INFO: [H_EST_NO] h_true={attenuation.real:.4f}{attenuation.imag:+.4f}j, h_est_no={h_est_no.real:.4f}{h_est_no.imag:+.4f}j")

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

# ---------------- Reusable simulation utilities for sweeps ----------------


def simulate_frame_ser(M=16, sps=8, fs=1e6, Nbits=4000,
                       sync_len_bits=26, ch_pilot_len_bits=128,
                       snr_db=10, with_awgn=True, rng=None):
    """
    Simulate a single random frame and return (num_symbol_errors, total_symbols).
    rng: instance of numpy.random.RandomState (if None uses global np.random)
    """
    if rng is None:
        rng = np.random

    # Build pieces
    sync_bits = _make_sync_bits(sync_len_bits)
    chpilot_bits = rng.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = rng.randint(0, M, size=(Nbits,))

    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps

    attenuation = (rng.rand() + 1j*rng.rand())
    attenuated_tx = attenuation * tx_frame

    random_index = rng.randint(1, frame_samples+1)
    idx0 = random_index - 1
    rx_no_noise = np.concatenate([attenuated_tx[idx0:], attenuated_tx[:idx0]])

    rx = awgn(rx_no_noise, snr_db) if with_awgn else rx_no_noise.copy()

    # Sync detect & pilot extraction
    rx2 = np.concatenate([rx, rx])
    corr = np.abs(np.convolve(rx2, np.conjugate(sync_shaped[::-1])))
    peak = np.argmax(corr)
    pilot_start_in_rx2 = peak - (len(sync_shaped) - 1)
    pilot_start = pilot_start_in_rx2 % frame_samples

    chpilot_start = pilot_start + len(sync_shaped)
    chpilot_indices = (np.arange(chpilot_start, chpilot_start +
                       ch_pilot_len_bits*sps) % frame_samples).astype(int)
    chpilot_samples = rx[chpilot_indices]
    chpilot_mat = chpilot_samples.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols = np.sum(chpilot_mat, axis=0).astype(np.complex128)

    h_est = np.mean(rx_chpilot_symbols / chpilot_symbols)

    # data extraction
    data_start = chpilot_start + ch_pilot_len_bits*sps
    data_indices = (np.arange(data_start, data_start + Nbits*sps) %
                    frame_samples).astype(int)
    data_samples = rx[data_indices]
    data_mat = data_samples.reshape((sps, Nbits), order='F')
    symbol_samples = np.sum(data_mat, axis=0).astype(np.complex128)

    # equalize, demodulate & count errors
    symbol_samples_eq = symbol_samples / h_est
    rx_syms = pskdemod(symbol_samples_eq, M)
    num_errors = np.sum(rx_syms != data_bits)
    return int(num_errors), int(Nbits)


def monte_carlo_ser(n_trials=100, seed=None, **simulate_kwargs):
    """
    Run n_trials independent frames and estimate SER (symbol error rate).
    Returns ser (float), total_errors (int), total_symbols (int)
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

# ---------------- Plot helpers for the five objectives ----------------


def plot_ser_vs_sps(fs, M, Nbits, sps_list, ch_pilot_len_bits=128,
                    sync_len_bits=26, snr_db=10, n_trials=80, with_awgn=True):
    """Objective (i): SER vs samples-per-symbol (sps). Also plot vs Rs = fs/sps."""
    sers = []
    for sps in sps_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits,
                                    snr_db=snr_db, with_awgn=with_awgn)
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
                         ch_pilot_len_bits=128, snr_db=10, n_trials=80, with_awgn=True):
    """Objective (ii): SER vs sync/channel-estimation sequence length."""
    sers = []
    for sync_len in sync_len_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len,
                                    snr_db=snr_db, with_awgn=with_awgn)
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
                  sync_len_bits=26, snr_db=10, n_trials=80, with_awgn=True):
    """Objective (iii): SER vs modulation order M (M-PSK)."""
    sers = []
    for M in M_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits,
                                    snr_db=snr_db, with_awgn=with_awgn)
        print(f"INFO: M={M} SER={ser:.4e}")
        sers.append(ser)
    plt.figure(figsize=(7, 5))
    plt.semilogy(M_list, sers, '-o')
    plt.title(f'SER vs Modulation order M (fs={fs}, N={Nbits}, sps={sps})')
    plt.xlabel('M (PSK order)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(M_list), np.array(sers)


def plot_ser_vs_N(fs, M, sps, N_list, ch_pilot_len_bits=128, sync_len_bits=26,
                  snr_db=10, n_trials=80, with_awgn=True):
    """Objective (iv): SER vs number of data symbols N."""
    sers = []
    for Nbits in N_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits,
                                    snr_db=snr_db, with_awgn=with_awgn)
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
                           ch_pilot_len_bits=128, sync_len_bits=26, snr_db=10,
                           n_trials=60, with_awgn=True, top_k=5):
    """
    Objective (v): coarse search across (sps, N, fs) for each M in M_list.
    Returns dict: M -> list of (ser, sps, N, fs, Rs)
    """
    results = {}
    for M in M_list:
        combos = []
        for sps in sps_candidates:
            for N in N_candidates:
                for fs in fs_candidates:
                    ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                                M=M, sps=sps, fs=fs, Nbits=N,
                                                ch_pilot_len_bits=ch_pilot_len_bits,
                                                sync_len_bits=sync_len_bits,
                                                snr_db=snr_db, with_awgn=with_awgn)
                    Rs = fs / sps
                    combos.append((ser, sps, N, fs, Rs))
                    print(
                        f"INFO: M={M} sps={sps} N={N} fs={fs:.2e} Rs={Rs:.2f} SER={ser:.4e}")
        combos.sort(key=lambda x: x[0])
        results[M] = combos[:top_k]
    return results


# ---------------- Example execution (single demo + quick plots) ----------------
if __name__ == "__main__":
    # 1) Run single-frame demo and show its constellation/correlation plots
    demo_results = run_single_frame_demo(
        Nbits=2000, M=16, sps=8, fs=1e6, snr_db=10, with_awgn=True, seed=1)

    # 2) Quick example sweeps for the five objectives (small n_trials for speed).
    #    Increase n_trials for higher statistical accuracy.
    fs = 1e6
    M = 16
    Nbits = 2000

    # Objective (i): SER vs sps
    sps_list = [1, 2, 4, 8, 16]
    plot_ser_vs_sps(fs=fs, M=M, Nbits=Nbits,
                    sps_list=sps_list, n_trials=60, snr_db=10)

    # Objective (ii): SER vs sync length
    sync_len_list = [8, 13, 26, 52]
    plot_ser_vs_sync_len(fs=fs, M=M, Nbits=Nbits,
                         sync_len_list=sync_len_list, sps=8, n_trials=60, snr_db=10)

    # Objective (iii): SER vs modulation order M
    plot_ser_vs_M(fs=fs, M_list=[2, 4, 8, 16, 32],
                  Nbits=1000, sps=8, n_trials=60, snr_db=10)

    # Objective (iv): SER vs number of data symbols N
    plot_ser_vs_N(fs=fs, M=M, sps=8, N_list=[
                  500, 1000, 2000, 4000], n_trials=60, snr_db=10)

    # Objective (v): coarse search for best params (keep candidate lists small)
    results = find_best_params_for_M(M_list=[4, 16], sps_candidates=[2, 4, 8], N_candidates=[
                                     1000, 2000], fs_candidates=[1e6, 2e6], n_trials=40, snr_db=10)
    print("\nBest combos (per M):")
    for Mkey, combos in results.items():
        print(f"M={Mkey}")
        for ser, sps_v, N_v, fs_v, Rs_v in combos:
            print(
                f"  SER={ser:.4e} sps={sps_v} N={N_v} fs={fs_v:.2e} Rs={Rs_v:.2f}")

    plt.show()
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------------------------------------------
# 🧾 Helper to parse list from user entry
# ------------------------------------------------------------
def parse_list(entry, default_list):
    text = entry.get().strip()
    if not text:
        return default_list
    try:
        # Convert comma-separated values to list of numbers
        vals = [int(float(x.strip())) for x in text.split(",")]
        return vals
    except Exception:
        messagebox.showerror("Invalid List", f"Could not parse: {text}")
        return default_list

# ------------------------------------------------------------
# 🧪 Run selected plots based on checkboxes
# ------------------------------------------------------------
def run_selected_plots():
    try:
        M = int(entry_M.get()) if entry_M.get() else 16
        Nbits = int(entry_Nbits.get()) if entry_Nbits.get() else 2000
        sps = int(float(entry_sps.get())) if entry_sps.get() else 8   # ✅ cast to int
        fs = int(float(entry_fs.get())) if entry_fs.get() else 1e6
        snr_db = int(float(entry_snr.get())) if entry_snr.get() else 10

    except ValueError:
        messagebox.showerror("Invalid Input", "Check the common simulation parameters.")
        return

    log_box.insert(tk.END, "▶ Running selected plots...\n")

    # Plot SER vs SPS
    if var_sps.get():
        sps_list = parse_list(entry_sps_list, [1, 2, 4, 8, 16])
        log_box.insert(tk.END, f"  - plot_ser_vs_sps with sps_list={sps_list}\n")
        plot_ser_vs_sps(fs=fs, M=M, Nbits=Nbits, sps_list=sps_list, n_trials=20, snr_db=snr_db)

    # Plot SER vs Sync Len
    if var_sync.get():
        sync_len_list = parse_list(entry_sync_list, [8, 13, 26, 52])
        log_box.insert(tk.END, f"  - plot_ser_vs_sync_len with sync_len_list={sync_len_list}\n")
        plot_ser_vs_sync_len(fs=fs, M=M, Nbits=Nbits, sync_len_list=sync_len_list, sps=sps, n_trials=20, snr_db=snr_db)

    # Plot SER vs M
    if var_mod.get():
        M_list = parse_list(entry_mod_list, [2, 4, 8, 16, 32])
        log_box.insert(tk.END, f"  - plot_ser_vs_M with M_list={M_list}\n")
        plot_ser_vs_M(fs=fs, M_list=M_list, Nbits=Nbits, sps=sps, n_trials=20, snr_db=snr_db)

    # Plot SER vs N
    if var_N.get():
        N_list = parse_list(entry_N_list, [500, 1000, 2000, 4000])
        log_box.insert(tk.END, f"  - plot_ser_vs_N with N_list={N_list}\n")
        plot_ser_vs_N(fs=fs, M=M, sps=sps, N_list=N_list, n_trials=20, snr_db=snr_db)

    # Best params search
    if var_best.get():
        M_list = parse_list(entry_best_M_list, [4, 16])
        sps_candidates = parse_list(entry_best_sps_list, [2, 4, 8])
        N_candidates = parse_list(entry_best_N_list, [1000, 2000])
        fs_candidates = parse_list(entry_best_fs_list, [1e6, 2e6])
        log_box.insert(tk.END, f"  - find_best_params_for_M with M_list={M_list}\n")
        results = find_best_params_for_M(
            M_list=M_list,
            sps_candidates=sps_candidates,
            N_candidates=N_candidates,
            fs_candidates=fs_candidates,
            n_trials=20,
            snr_db=snr_db,
        )
        for Mkey, combos in results.items():
            log_box.insert(tk.END, f"   Best for M={Mkey}:\n")
            for ser, sps_v, N_v, fs_v, Rs_v in combos:
                log_box.insert(tk.END, f"     SER={ser:.4e}, sps={sps_v}, N={N_v}, fs={fs_v:.2e}, Rs={Rs_v:.2f}\n")

    log_box.insert(tk.END, "✅ Selected plots completed.\n")
    log_box.see(tk.END)
    plt.show()

# ------------------------------------------------------------
# 🧰 Dynamic toggle of plot parameter fields
# ------------------------------------------------------------
def toggle_entry(var, entry):
    if var.get():
        entry.configure(state="normal")
    else:
        entry.configure(state="disabled")

# ------------------------------------------------------------
# 🖼 UI Layout
# ------------------------------------------------------------
def launch_ui():
    global entry_M, entry_Nbits, entry_sps, entry_fs, entry_snr
    global entry_sps_list, entry_sync_list, entry_mod_list, entry_N_list
    global entry_best_M_list, entry_best_sps_list, entry_best_N_list, entry_best_fs_list
    global var_sps, var_sync, var_mod, var_N, var_best, log_box

    root = tk.Tk()
    root.title("Channel Estimation — Selective Plot UI")
    root.geometry("650x600")

    # Common parameters
    frame_common = ttk.LabelFrame(root, text="Common Simulation Parameters")
    frame_common.pack(fill="x", padx=10, pady=10)

    def make_row(label, row):
        ttk.Label(frame_common, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(frame_common)
        entry.grid(row=row, column=1, padx=5, pady=5)
        return entry

    entry_M = make_row("M (modulation):", 0)
    entry_Nbits = make_row("Nbits (bits):", 1)
    entry_sps = make_row("sps (samples/sym):", 2)
    entry_fs = make_row("fs (Hz):", 3)
    entry_snr = make_row("SNR (dB):", 4)

    # Plot selection
    frame_plots = ttk.LabelFrame(root, text="Select Plots to Run")
    frame_plots.pack(fill="x", padx=10, pady=10)

    def make_plot_row(var, text, default, row):
        cb = ttk.Checkbutton(frame_plots, text=text, variable=var, command=lambda: toggle_entry(var, default))
        cb.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        default.grid(row=row, column=1, padx=5, pady=5)
        default.configure(state="disabled")

    var_sps = tk.BooleanVar()
    entry_sps_list = ttk.Entry(frame_plots)
    entry_sps_list.insert(0, "1,2,4,8,16")
    make_plot_row(var_sps, "plot_ser_vs_sps", entry_sps_list, 0)

    var_sync = tk.BooleanVar()
    entry_sync_list = ttk.Entry(frame_plots)
    entry_sync_list.insert(0, "8,13,26,52")
    make_plot_row(var_sync, "plot_ser_vs_sync_len", entry_sync_list, 1)

    var_mod = tk.BooleanVar()
    entry_mod_list = ttk.Entry(frame_plots)
    entry_mod_list.insert(0, "2,4,8,16,32")
    make_plot_row(var_mod, "plot_ser_vs_M", entry_mod_list, 2)

    var_N = tk.BooleanVar()
    entry_N_list = ttk.Entry(frame_plots)
    entry_N_list.insert(0, "500,1000,2000,4000")
    make_plot_row(var_N, "plot_ser_vs_N", entry_N_list, 3)

    var_best = tk.BooleanVar()
    entry_best_M_list = ttk.Entry(frame_plots)
    entry_best_M_list.insert(0, "4,16")
    entry_best_sps_list = ttk.Entry(frame_plots)
    entry_best_sps_list.insert(0, "2,4,8")
    entry_best_N_list = ttk.Entry(frame_plots)
    entry_best_N_list.insert(0, "1000,2000")
    entry_best_fs_list = ttk.Entry(frame_plots)
    entry_best_fs_list.insert(0, "1e6,2e6")

    cb_best = ttk.Checkbutton(frame_plots, text="find_best_params_for_M", variable=var_best,
                              command=lambda: [toggle_entry(var_best, e) for e in (entry_best_M_list, entry_best_sps_list, entry_best_N_list, entry_best_fs_list)])
    cb_best.grid(row=4, column=0, sticky="w", padx=5, pady=5)

    entry_best_M_list.grid(row=4, column=1, padx=5, pady=2)
    entry_best_sps_list.grid(row=5, column=1, padx=5, pady=2)
    entry_best_N_list.grid(row=6, column=1, padx=5, pady=2)
    entry_best_fs_list.grid(row=7, column=1, padx=5, pady=2)

    for e in (entry_best_M_list, entry_best_sps_list, entry_best_N_list, entry_best_fs_list):
        e.configure(state="disabled")

    # Run Button
    btn_run = ttk.Button(root, text="▶ Run Selected Plots", command=run_selected_plots)
    btn_run.pack(pady=10)

    # Log box
    frame_log = ttk.LabelFrame(root, text="Log / Status")
    frame_log.pack(fill="both", expand=True, padx=10, pady=10)
    log_box = tk.Text(frame_log, wrap="word", height=10)
    log_box.pack(fill="both", expand=True)

    root.mainloop()

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    launch_ui()
