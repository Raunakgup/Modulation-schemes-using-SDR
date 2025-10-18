"""
PSK frame simulation & analysis (refactored + heavily commented)

This module simulates frames made of:
  [sync sequence] [channel pilot] [data symbols]
where each symbol is NRZ-shaped (repeated sps times), the frame is
circularly-shifted (to emulate an SDR buffer), passed through a
random complex scalar channel, optionally AWGN is added, and then
synchronization + pilot-based channel estimation + equalization is
performed to recover data symbols.

Main highlights / improvements in this refactor:
- Added type hints and thorough docstrings for clarity.
- Added extensive inline comments explaining each step.
- Added small robustness checks (avoid divide-by-zero on channel estimate).
- Improved debug printing in `print_snip` (shape, dtype, short sample list).
- Kept original algorithms (correlation-based sync, pilot averaging).
- Preserved plotting behaviors in run_single_frame_demo.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Reproducible randomness when the module is imported/run interactively.
# Individual functions accept seed/rng parameters for repeatability.
np.random.seed(1)


# ---------------- Utility helpers ----------------

def print_snip(name: str, x: Optional[Iterable[Any]]) -> None:
    """
    Print a short diagnostic summary of an array-like value:
      - name
      - dtype and shape
      - first up to 12 elements (formatted compactly)
    This helps when debugging simulation flows without printing huge arrays.
    """
    if x is None:
        return
    x_arr = np.asarray(x)
    N = min(12, x_arr.size)
    xs = x_arr.ravel()[:N]

    def fmt(v: Any) -> str:
        # format complex numbers consistently and real numbers simply
        v = np.asarray(v)
        if np.iscomplexobj(v):
            re = float(np.real(v))
            im = float(np.imag(v))
            return f"{re:.4g}{im:+.4g}j"
        else:
            return f"{float(v):.4g}"

    sample_str = ", ".join(fmt(v) for v in xs)
    print(f"INFO: [{name}] dtype={x_arr.dtype}, shape={x_arr.shape}, first={{ {sample_str} }}")



def pskmod(symbols: Iterable[int], M: int = 2) -> np.ndarray:
    """
    General M-PSK modulator.
    Input:
      - symbols: integer symbols (array-like) expected nominally in 0..M-1.
                 Values are reduced modulo M internally for safety.
      - M: PSK order, e.g. 2,4,8,16...
    Output:
      - complex constellation points on the unit circle:
        exp(1j * 2*pi * k / M)
    Notes:
      - For non-binary inputs (e.g. sync bits with values 0/1), those values map
        to the first M constellation points (so 0->angle 0, 1->angle 2*pi/M).
      - Output type is complex128 for numerical stability.
    """
    syms = np.asarray(symbols).astype(int).ravel()
    syms = np.mod(syms, M)  # wrap around M to be robust to inputs outside 0..M-1
    const = np.exp(1j * 2.0 * np.pi * syms / M)
    return const.astype(np.complex128)


def pskdemod(symbols: Iterable[complex], M: int = 2) -> np.ndarray:
    """
    M-PSK demodulation by minimum Euclidean distance to the M constellation points.
    Returns integer decisions in 0..M-1 for each input complex sample.
    """
    symbols_arr = np.asarray(symbols).ravel().astype(np.complex128)
    # Precompute the M constellation reference points on the unit circle:
    const = np.exp(1j * 2.0 * np.pi * np.arange(M) / M).astype(np.complex128)
    # Compute pairwise differences and choose the nearest constellation point.
    # This is vectorized: diff shape => (num_symbols, M)
    diff = symbols_arr.reshape(-1, 1) - const.reshape(1, -1)
    dist = np.abs(diff)  # Euclidean distance for each candidate
    decisions = np.argmin(dist, axis=1).astype(int)
    return decisions


def awgn(sig: Iterable[complex], snr_db: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Additive White Gaussian Noise (complex) to a complex signal `sig`.
    - sig: complex array
    - snr_db: desired SNR in dB (signal power to noise power)
    - rng: optional numpy RandomState for reproducibility (uses global np.random if None)
    Returns the noisy signal (complex128).
    """
    if rng is None:
        # use numpy's global RNG (np.random)
        randn = np.random.randn
    else:
        randn = rng.randn

    sig = np.asarray(sig)
    # signal power (average of |s|^2)
    power_signal = np.mean(np.abs(sig)**2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power_signal / snr_linear
    # for complex noise: real and imaginary parts each have variance = noise_power/2
    noise = np.sqrt(noise_power / 2.0) * (randn(*sig.shape) + 1j * randn(*sig.shape))
    return (sig + noise).astype(np.complex128)


def biterr(a: Iterable[int], b: Iterable[int]) -> Tuple[int, float]:
    """
    Count integer mismatches between two arrays a and b (converted to flattened int arrays).
    Returns (num_errors, bit_error_rate).
    Note: In this code we use symbol errors (not bit-level Hamming) because
    data_bits are symbol indexes (0..M-1). For clarity we call it 'biterr' to match your original.
    """
    a_arr = np.asarray(a).ravel().astype(int)
    b_arr = np.asarray(b).ravel().astype(int)
    if a_arr.size != b_arr.size:
        raise ValueError("biterr: inputs must have the same total size")
    errs = int(np.sum(a_arr != b_arr))
    ber = float(errs) / float(a_arr.size) if a_arr.size > 0 else float('nan')
    return errs, ber


# ---------------- Sync helper and default Barker ----------------
DEFAULT_BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=int)


def _make_sync_bits(sync_len_bits: int, barker: np.ndarray = DEFAULT_BARKER13) -> np.ndarray:
    """
    Produce a binary sync bit sequence of length `sync_len_bits` by repeating
    the provided Barker sequence as necessary and converting -1->1, +1->0 convention:
      - the original Barker array contains +1 and -1 entries
      - we map (barker < 0) to 1, others to 0
    This mirrors the original code's behavior and lets the sync be used with
    general M-PSK modulators (where sync bits are 0/1).
    """
    reps = ceil(sync_len_bits / len(barker))
    seq = np.tile(barker, reps)[:sync_len_bits]
    # convert -1 (negative) to 1 and +1 to 0 to get a binary sequence
    return (seq < 0).astype(int)


# ---------------- Single-frame demo (with plotting) ----------------

def run_single_frame_demo(
    Nbits: int = 4000,
    M: int = 2,
    sps: int = 8,
    fs: float = 1e6,
    snr_db: float = 10,
    sync_barker13: Optional[np.ndarray] = None,
    ch_pilot_len_bits: int = 128,
    with_awgn: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single-frame demonstration of:
      1) build frame = [sync_shaped, pilot_shaped, data_shaped]
      2) pass through a random scalar complex channel
      3) circularly shift to emulate SDR buffer wrap-around
      4) optionally add AWGN
      5) perform sync detection by correlating with the known sync_shaped
      6) extract pilot, estimate channel by averaging pilot symbols
      7) extract data, equalize via division by estimated channel
      8) demodulate before/after equalization and report BER
      9) produce a set of constellation & correlation plots

    Returns a dictionary containing BER numbers and several arrays useful for
    further analysis (e.g. symbol_samples_eq_aw).
    """
    # allow reproducible runs per-call
    if seed is not None:
        np.random.seed(seed)

    # Use default Barker if none provided. The code duplicates the Barker 3x
    # for robustness (this matches the original behavior).
    if sync_barker13 is None:
        sync_barker13 = DEFAULT_BARKER13
    sync_barker13 = np.tile(sync_barker13, 3)  # repetition for robustness
    sync_len_bits = len(sync_barker13)

    print(f"INFO: [PARAM] Nbits={Nbits}, M={M}, sps={sps}, sync_len={sync_len_bits}, chpilot_len={ch_pilot_len_bits}, snr_db={snr_db}")

    # ---------------- Build transmit pieces ----------------
    # Sync/pilot/data are expressed as integer symbols before modulation.
    # Note: sync bits are binary (0/1) and will map to PSK points 0 and 1.
    sync_bits = (sync_barker13 < 0).astype(int)
    chpilot_bits = np.random.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = np.random.randint(0, M, size=(Nbits,))

    # Map integer symbols -> complex constellation points
    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    # Debug prints for initial symbol sequences
    print_snip('sync_bits', sync_bits)
    print_snip('sync_symbols', sync_symbols)
    print_snip('chpilot_bits', chpilot_bits)
    print_snip('chpilot_symbols', chpilot_symbols)
    print_snip('data_bits (first)', data_bits[:min(12, data_bits.size)])
    print_snip('data_symbols (first)', data_symbols[:min(12, data_symbols.size)])

    # ---------------- NRZ shaping ----------------
    # Each symbol is repeated 'sps' times (simple rectangular pulse shaping).
    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    print(f"INFO: [SHAPE] sizes: sync_shaped={len(sync_shaped)}, chpilot_shaped={len(chpilot_shaped)}, data_shaped={len(data_shaped)}")
    print_snip('sync_shaped (first samples)', sync_shaped[:min(12, sync_shaped.size)])

    # Concatenate to form the transmit frame in sample domain
    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps
    print(f"INFO: [TX] frame_bits={frame_bits}, frame_samples={frame_samples}")
    print_snip('tx_frame (first)', tx_frame[:min(12, tx_frame.size)])

    # ---------------- Channel: random complex scalar ----------------
    attenuation = (np.random.rand() + 1j * np.random.rand())
    print(f"INFO: [CH] True attenuation = {attenuation.real:.4f}{attenuation.imag:+.4f}j")
    attenuated_tx = attenuation * tx_frame

    # ---------------- SDR-like circular shift (wrap) ----------------
    # Pick a random shift index in 1..frame_samples inclusive (matching original)
    random_index = np.random.randint(1, frame_samples + 1)
    idx0 = random_index - 1
    # Circular buffer wrap (concatenate tail then head)
    rx_no_noise = np.concatenate([attenuated_tx[idx0:], attenuated_tx[:idx0]])
    print(f"INFO: [RX_SHIFT] random_index={random_index}")
    print_snip('rx_no_noise (first)', rx_no_noise[:min(12, rx_no_noise.size)])

    # ---------------- AWGN ----------------
    rx_awgn = awgn(rx_no_noise, snr_db) if with_awgn else rx_no_noise.copy()
    print(f"INFO: [AWGN] Applied AWGN at {snr_db} dB (with_awgn={with_awgn})")
    print_snip('rx_awgn (first)', rx_awgn[:min(12, rx_awgn.size)])

    # ---------------- Correlation & sync detection (no AWGN) ----------------
    # To avoid wrap issues we correlate with rx concatenated to itself.
    rx2_no = np.concatenate([rx_no_noise, rx_no_noise])
    # correlate with conjugated reversed sync_shaped (matched filter)
    corr_no = np.abs(np.convolve(rx2_no, np.conjugate(sync_shaped[::-1])))
    peak_no = np.argmax(corr_no)
    # compute pilot start (index within doubled array), then map into single frame
    pilot_start_in_rx2_no = peak_no - (len(sync_shaped) - 1)
    pilot_start_no = pilot_start_in_rx2_no % frame_samples
    # print indices using 1-based position in human-friendly manner
    print(f"INFO: [SYNC_NO] peak_no={peak_no}, pilot_start_no={pilot_start_no+1}, pilot_start mod sps={(pilot_start_no % sps)+1}")

    # Check match fraction for the sync segment (exact-match tolerance small)
    seglen_no = min(len(sync_shaped), frame_samples - pilot_start_no)
    seg_no = rx_no_noise[pilot_start_no: (pilot_start_no + seglen_no)]
    match_frac_no = np.sum(np.abs(seg_no - sync_shaped[:seglen_no]) < 1e-6) / seglen_no
    print(f"INFO: [SYNC_MATCH_NO] seglen={seglen_no}, match_frac={match_frac_no:.4f}")

    # ---------------- Channel pilot extraction (no AWGN) ----------------
    chpilot_start_no = pilot_start_no + len(sync_shaped)
    # compute indices into the circular frame (wrap using modulo)
    chpilot_indices_no = (np.arange(chpilot_start_no, chpilot_start_no + ch_pilot_len_bits * sps) % frame_samples).astype(int)
    chpilot_samples_no = rx_no_noise[chpilot_indices_no]
    # reshape into (sps, ch_pilot_len_bits) column-major (Fortran) so that
    # each column corresponds to the sps samples for one pilot symbol.
    chpilot_mat_no = chpilot_samples_no.reshape((sps, ch_pilot_len_bits), order='F')
    # sum across the sps samples to form one complex sample per symbol (simple integrate-and-dump)
    rx_chpilot_symbols_no = np.sum(chpilot_mat_no, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_no (first)', rx_chpilot_symbols_no[:min(12, rx_chpilot_symbols_no.size)])

    # Estimate channel by averaging ratio rx/pilot (scalar channel)
    # add small epsilon in denominator if pilot symbol is zero (shouldn't happen for PSK)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_est_no = np.mean(rx_chpilot_symbols_no / chpilot_symbols)
    # safety: protect against an accidental (almost) zero estimate
    if np.abs(h_est_no) < 1e-12:
        print("WARN: [H_EST_NO] small channel estimate magnitude; replacing with small non-zero value to avoid divide-by-zero.")
        h_est_no = 1e-12 + 0j
    print(f"INFO: [H_EST_NO] h_true={attenuation.real:.4f}{attenuation.imag:+.4f}j, h_est_no={h_est_no.real:.4f}{h_est_no.imag:+.4f}j")

    # ---------------- Extract data symbols (before equalization) ----------------
    data_start_no = chpilot_start_no + ch_pilot_len_bits * sps
    data_indices_no = (np.arange(data_start_no, data_start_no + Nbits * sps) % frame_samples).astype(int)
    data_samples_no = rx_no_noise[data_indices_no]
    data_mat_no = data_samples_no.reshape((sps, Nbits), order='F')
    # integrate-and-dump to one complex sample per data symbol
    symbol_samples_no = np.sum(data_mat_no, axis=0).astype(np.complex128)
    print_snip('symbol_samples_no (first)', symbol_samples_no[:min(12, symbol_samples_no.size)])

    # ---------------- Demodulate BEFORE equalization (no AWGN) ----------------
    rx_syms_before_no = pskdemod(symbol_samples_no, M)
    errs_before_no, ber_before_no = biterr(data_bits, rx_syms_before_no)
    print(f"INFO: [DEM_BEFORE_NO] NumErr={errs_before_no}, BER={ber_before_no:.6g}")

    # ---------------- Equalize (no AWGN) + demodulate ----------------
    symbol_samples_eq_no = symbol_samples_no / h_est_no
    print_snip('symbol_samples_eq_no (first)', symbol_samples_eq_no[:min(12, symbol_samples_eq_no.size)])
    rx_syms_after_no = pskdemod(symbol_samples_eq_no, M)
    errs_after_no, ber_after_no = biterr(data_bits, rx_syms_after_no)
    print(f"INFO: [DEM_AFTER_NO] NumErr={errs_after_no}, BER={ber_after_no:.6g}")

    # ---------------- Now repeat the same with AWGN (perform sync & pilot extraction on noisy rx) ----------------
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
    chpilot_indices_aw = (np.arange(chpilot_start_aw, chpilot_start_aw + ch_pilot_len_bits * sps) % frame_samples).astype(int)
    chpilot_samples_aw = rx_awgn[chpilot_indices_aw]
    chpilot_mat_aw = chpilot_samples_aw.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols_aw = np.sum(chpilot_mat_aw, axis=0).astype(np.complex128)
    print_snip('rx_chpilot_symbols_aw (first)', rx_chpilot_symbols_aw[:min(12, rx_chpilot_symbols_aw.size)])

    with np.errstate(divide='ignore', invalid='ignore'):
        h_est_aw = np.mean(rx_chpilot_symbols_aw / chpilot_symbols)
    if np.abs(h_est_aw) < 1e-12:
        print("WARN: [H_EST_AW] small channel estimate magnitude; replacing with small non-zero value.")
        h_est_aw = 1e-12 + 0j
    print(f"INFO: [H_EST_AW] h_est_aw={h_est_aw.real:.4f}{h_est_aw.imag:+.4f}j")

    # ---------------- Extract and demodulate data under AWGN ----------------
    data_start_aw = chpilot_start_aw + ch_pilot_len_bits * sps
    data_indices_aw = (np.arange(data_start_aw, data_start_aw + Nbits * sps) % frame_samples).astype(int)
    data_samples_aw = rx_awgn[data_indices_aw]
    data_mat_aw = data_samples_aw.reshape((sps, Nbits), order='F')
    symbol_samples_aw = np.sum(data_mat_aw, axis=0).astype(np.complex128)
    print_snip('symbol_samples_aw (first)', symbol_samples_aw[:min(12, symbol_samples_aw.size)])

    rx_syms_before_aw = pskdemod(symbol_samples_aw, M)
    errs_before_aw, ber_before_aw = biterr(data_bits, rx_syms_before_aw)
    print(f"INFO: [DEM_BEFORE_AW] NumErr={errs_before_aw}, BER={ber_before_aw:.6g}")

    symbol_samples_eq_aw = symbol_samples_aw / h_est_aw
    print_snip('symbol_samples_eq_aw (first)', symbol_samples_eq_aw[:min(12, symbol_samples_eq_aw.size)])
    rx_syms_after_aw = pskdemod(symbol_samples_eq_aw, M)
    errs_after_aw, ber_after_aw = biterr(data_bits, rx_syms_after_aw)
    print(f"INFO: [DEM_AFTER_AW] NumErr={errs_after_aw}, BER={ber_after_aw:.6g}")

    # ---------------- Plots (constellations, correlations) ----------------
    # The plots show before/after equalization for both no-AWGN and with-AWGN cases.
    plt.figure('No AWGN: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_no[:2000]), np.imag(symbol_samples_no[:2000]), '.', markersize=6)
    plt.title('No AWGN — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_no[:2000]), np.imag(symbol_samples_eq_no[:2000]), '.', markersize=6)
    plt.title('No AWGN — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.figure('With AWGN: Before/After Equalization', figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(symbol_samples_aw[:2000]), np.imag(symbol_samples_aw[:2000]), '.')
    plt.title(f'AWGN {snr_db} dB — before equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(symbol_samples_eq_aw[:2000]), np.imag(symbol_samples_eq_aw[:2000]), '.')
    plt.title(f'AWGN {snr_db} dB — after equalization (subset)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.axis('equal')
    plt.grid(True)

    # Correlation peaks (noisy and noiseless)
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

    print(f"\nINFO: [SUMMARY] NO_AWGN: BER_before={ber_before_no:.6g} BER_after={ber_after_no:.6g} | WITH_AWGN: BER_before={ber_before_aw:.6g} BER_after={ber_after_aw:.6g}")

    # Return useful artifacts for further inspection/tests
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


# ---------------- Reusable simulation utilities for Monte-Carlo sweeps ----------------

def simulate_frame_ser(
    M: int = 16,
    sps: int = 8,
    fs: float = 1e6,
    Nbits: int = 4000,
    sync_len_bits: int = 26,
    ch_pilot_len_bits: int = 128,
    snr_db: float = 10,
    with_awgn: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[int, int]:
    """
    Simulate a single random frame and return (num_symbol_errors, total_symbols).
    - rng: numpy.random.RandomState instance for reproducibility (if None, uses global np.random)
    """
    if rng is None:
        rng = np.random

    # Build data pieces (binary sync from _make_sync_bits)
    sync_bits = _make_sync_bits(sync_len_bits)
    chpilot_bits = rng.randint(0, M, size=(ch_pilot_len_bits,))
    data_bits = rng.randint(0, M, size=(Nbits,))

    sync_symbols = pskmod(sync_bits, M)
    chpilot_symbols = pskmod(chpilot_bits, M)
    data_symbols = pskmod(data_bits, M)

    # NRZ shaping
    sync_shaped = np.repeat(sync_symbols, sps)
    chpilot_shaped = np.repeat(chpilot_symbols, sps)
    data_shaped = np.repeat(data_symbols, sps)

    tx_frame = np.concatenate([sync_shaped, chpilot_shaped, data_shaped])
    frame_bits = sync_len_bits + ch_pilot_len_bits + Nbits
    frame_samples = frame_bits * sps

    # Random scalar channel, same distribution as run_single_frame_demo
    attenuation = (rng.rand() + 1j * rng.rand())
    attenuated_tx = attenuation * tx_frame

    # Circular shift
    random_index = rng.randint(1, frame_samples + 1)
    idx0 = random_index - 1
    rx_no_noise = np.concatenate([attenuated_tx[idx0:], attenuated_tx[:idx0]])

    rx = awgn(rx_no_noise, snr_db, rng) if with_awgn else rx_no_noise.copy()

    # Sync detect and pilot extraction
    rx2 = np.concatenate([rx, rx])
    corr = np.abs(np.convolve(rx2, np.conjugate(sync_shaped[::-1])))
    peak = np.argmax(corr)
    pilot_start_in_rx2 = peak - (len(sync_shaped) - 1)
    pilot_start = pilot_start_in_rx2 % frame_samples

    chpilot_start = pilot_start + len(sync_shaped)
    chpilot_indices = (np.arange(chpilot_start, chpilot_start + ch_pilot_len_bits * sps) % frame_samples).astype(int)
    chpilot_samples = rx[chpilot_indices]
    chpilot_mat = chpilot_samples.reshape((sps, ch_pilot_len_bits), order='F')
    rx_chpilot_symbols = np.sum(chpilot_mat, axis=0).astype(np.complex128)

    with np.errstate(divide='ignore', invalid='ignore'):
        h_est = np.mean(rx_chpilot_symbols / chpilot_symbols)
    if np.abs(h_est) < 1e-12:
        # avoid division by zero in equalization
        h_est = 1e-12 + 0j

    # Data extraction
    data_start = chpilot_start + ch_pilot_len_bits * sps
    data_indices = (np.arange(data_start, data_start + Nbits * sps) % frame_samples).astype(int)
    data_samples = rx[data_indices]
    data_mat = data_samples.reshape((sps, Nbits), order='F')
    symbol_samples = np.sum(data_mat, axis=0).astype(np.complex128)

    # Equalize, demodulate & count errors (symbol errors)
    symbol_samples_eq = symbol_samples / h_est
    rx_syms = pskdemod(symbol_samples_eq, M)
    num_errors = int(np.sum(rx_syms != data_bits))
    return num_errors, int(Nbits)


def monte_carlo_ser(n_trials: int = 100, seed: Optional[int] = None, **simulate_kwargs) -> Tuple[float, int, int]:
    """
    Run `n_trials` independent frames and estimate the symbol error rate (SER).
    Returns:
      - ser: errors / total_symbols (float)
      - total_errors: int
      - total_symbols: int
    Example usage:
      ser, total_err, total_sym = monte_carlo_ser(n_trials=200, seed=42, M=16, sps=8, ...)
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
    ser = (total_err / total_sym) if total_sym > 0 else float('nan')
    return ser, total_err, total_sym


# ---------------- Plot helpers for parameter sweeps ----------------

def plot_ser_vs_sps(fs: float, M: int, Nbits: int, sps_list: Iterable[int], ch_pilot_len_bits: int = 128,
                    sync_len_bits: int = 26, snr_db: float = 10, n_trials: int = 80, with_awgn: bool = True):
    """
    Objective (i): sweep samples-per-symbol (sps) and plot SER vs sps and vs symbol rate Rs = fs/sps.
    Returns arrays (sps_arr, Rs_arr, sers_arr).
    """
    sers: List[float] = []
    for sps in sps_list:
        ser, _, _ = monte_carlo_ser(n_trials=n_trials,
                                    M=M, sps=sps, fs=fs, Nbits=Nbits,
                                    ch_pilot_len_bits=ch_pilot_len_bits,
                                    sync_len_bits=sync_len_bits,
                                    snr_db=snr_db, with_awgn=with_awgn)
        print(f"INFO: sps={sps} SER={ser:.4e}")
        sers.append(ser)
    sps_arr = np.array(list(sps_list))
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


def plot_ser_vs_sync_len(fs: float, M: int, Nbits: int, sync_len_list: Iterable[int], sps: int = 8,
                         ch_pilot_len_bits: int = 128, snr_db: float = 10, n_trials: int = 80, with_awgn: bool = True):
    """
    Objective (ii): sweep sync sequence length used for synchronization/pilot and plot SER.
    """
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
    plt.semilogy(list(sync_len_list), sers, '-o')
    plt.title(f'SER vs sync length (fs={fs}, M={M}, sps={sps}, N={Nbits})')
    plt.xlabel('sync / pilot length (symbols)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(list(sync_len_list)), np.array(sers)


def plot_ser_vs_M(fs: float, M_list: Iterable[int], Nbits: int, sps: int = 8, ch_pilot_len_bits: int = 128,
                  sync_len_bits: int = 26, snr_db: float = 10, n_trials: int = 80, with_awgn: bool = True):
    """
    Objective (iii): SER vs modulation order M (M-PSK).
    """
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
    plt.semilogy(list(M_list), sers, '-o')
    plt.title(f'SER vs Modulation order M (fs={fs}, N={Nbits}, sps={sps})')
    plt.xlabel('M (PSK order)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(list(M_list)), np.array(sers)


def plot_ser_vs_N(fs: float, M: int, sps: int, N_list: Iterable[int], ch_pilot_len_bits: int = 128, sync_len_bits: int = 26,
                  snr_db: float = 10, n_trials: int = 80, with_awgn: bool = True):
    """
    Objective (iv): SER vs number of data symbols N (frame length).
    """
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
    plt.semilogy(list(N_list), sers, '-o')
    plt.title(f'SER vs Number of data symbols N (fs={fs}, M={M}, sps={sps})')
    plt.xlabel('Number of data symbols (N)')
    plt.ylabel('SER (log scale)')
    plt.grid(True)
    plt.show()
    return np.array(list(N_list)), np.array(sers)


def find_best_params_for_M(
    M_list: Iterable[int],
    sps_candidates: Iterable[int],
    N_candidates: Iterable[int],
    fs_candidates: Iterable[float],
    ch_pilot_len_bits: int = 128,
    sync_len_bits: int = 26,
    snr_db: float = 10,
    n_trials: int = 60,
    with_awgn: bool = True,
    top_k: int = 5,
) -> Dict[int, List[Tuple[float, int, int, float, float]]]:
    """
    Objective (v): coarse search across (sps, N, fs) for each M in M_list.
    Returns dict mapping M -> list of (ser, sps, N, fs, Rs) sorted by ascending SER (best first).
    """
    results: Dict[int, List[Tuple[float, int, int, float, float]]] = {}
    for M in M_list:
        combos: List[Tuple[float, int, int, float, float]] = []
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
                    print(f"INFO: M={M} sps={sps} N={N} fs={fs:.2e} Rs={Rs:.2f} SER={ser:.4e}")
        combos.sort(key=lambda x: x[0])  # sort by ser (ascending)
        results[M] = combos[:top_k]
    return results


# ---------------- Example execution when running the file directly ----------------
if __name__ == "__main__":
    # 1) Run a single-frame demo and display constellation/correlation plots.
    demo_results = run_single_frame_demo(Nbits=2000, M=16, sps=8, fs=1e6, snr_db=10, with_awgn=True, seed=1)

    # 2) Quick example parameter sweeps (use small n_trials here; increase for more accurate stats)
    fs = 1e6
    M = 16
    Nbits = 2000

    # Objective (i): SER vs sps
    sps_list = [1, 2, 4, 8, 16]
    plot_ser_vs_sps(fs=fs, M=M, Nbits=Nbits, sps_list=sps_list, n_trials=60, snr_db=10)

    # Objective (ii): SER vs sync length
    sync_len_list = [8, 13, 26, 52]
    plot_ser_vs_sync_len(fs=fs, M=M, Nbits=Nbits, sync_len_list=sync_len_list, sps=8, n_trials=60, snr_db=10)

    # Objective (iii): SER vs modulation order (M)
    plot_ser_vs_M(fs=fs, M_list=[2, 4, 8, 16, 32], Nbits=1000, sps=8, n_trials=60, snr_db=10)

    # Objective (iv): SER vs number of symbols N
    plot_ser_vs_N(fs=fs, M=M, sps=8, N_list=[500, 1000, 2000, 4000], n_trials=60, snr_db=10)

    # Objective (v): coarse parameter search (keep candidate lists small for speed)
    results = find_best_params_for_M(M_list=[4, 16], sps_candidates=[2, 4, 8], N_candidates=[1000, 2000], fs_candidates=[1e6, 2e6], n_trials=40, snr_db=10)
    print("\nBest combos (per M):")
    for Mkey, combos in results.items():
        print(f"M={Mkey}")
        for ser, sps_v, N_v, fs_v, Rs_v in combos:
            print(f"  SER={ser:.4e} sps={sps_v} N={N_v} fs={fs_v:.2e} Rs={Rs_v:.2f}")

    plt.show()
