
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import rir_generator as rir
import os
import sys

# --- Constants & Configuration ---
ROOM_DIMS = [5.2, 6.2, 3.5]  # meters [x, y, z]
FS = 16000  # Sampling rate (Hz) - typical for LibriSpeech
C = 343.0  # Speed of sound (m/s)
MIC_CENTER = np.array([2.6, 3.0, 1.5])
MIC_D = 0.2  # spacing
# 4 mics configuration: [x,y,z]
MICS = np.array([
    [2.6 - MIC_D / 2, 3.0, 1.5],
    [2.6 + MIC_D / 2, 3.0, 1.5],
    [2.6, 3.0 - MIC_D / 2, 1.5],
    [2.6, 3.0 + MIC_D / 2, 1.5]
])

SOURCE_X_RANGE = [1, 4]
SOURCE_Y_RANGE = [1, 5]
SOURCE_Z = 1.5  # Assumed same height as mics if not specified, or random? 
# "The source location is in a rectangular area... 1-4m x, 1-5m y". 
# Z is usually height of mouth, let's assume 1.5m (same plane) or random.
# Questions didn't specify Z range, but typically 2D loc on a plane or 3D.
# Given "rectangular domain" usually implies 2D search on a slice.
# Let's fix Source Z to 1.5m for simplicity unless specified otherwise.

# Grid for Heatmaps (20x20)
GRID_RES = 20
X_GRID = np.linspace(SOURCE_X_RANGE[0], SOURCE_X_RANGE[1], GRID_RES)
Y_GRID = np.linspace(SOURCE_Y_RANGE[0], SOURCE_Y_RANGE[1], GRID_RES)

# FFT parameters
N_FFT = 512
HOP_LEN = 256


# --- Helper Functions ---

def load_audio(filepath, target_len_sec=2.0):
    """Load and trim/pad audio to target length."""
    rate, data = wavfile.read(filepath)
    if rate != FS:
        # Simple resampling if needed, but assuming 16k based on context
        pass 
    
    # Take a snippet
    nsamples = int(target_len_sec * FS)
    if len(data) > nsamples:
        data = data[:nsamples]
    else:
        pad = nsamples - len(data)
        data = np.pad(data, (0, pad))
    
    # Normalize
    data = data.astype(np.float32)
    data /= np.max(np.abs(data)) + 1e-8
    return data

def generate_signals(clean_signal, source_pos, room_dims, mics, t60, snr_db):
    """
    Generate microphone signals using rir-generator and add noise.
    """
    # RIR generator expects C style code or specific wrapper
    # Using the python wrapper 'rir-generator'
    # rir.generate(c, fs, r, s, L, reverberation_time=T60, nsample=...)
    
    nsample = int(t60 * FS) if t60 > 0 else 4096 # Length of IR
    
    # Note: rir_generator args:
    # c: sound velocity
    # fs: sampling frequency
    # r: receiver positions (N_mics x 3)
    # s: source position (1 x 3)
    # L: Room dimensions (3)
    # reverberation_time: T60
    # nsample: number of samples to calculate
    
    h = rir.generate(
        c=C,
        fs=FS,
        r=mics,
        s=source_pos,
        L=room_dims,
        reverberation_time=t60,
        nsample=nsample
    )
    
    # Convolve
    # clean_signal is (N,)
    # h is (N_mics, Len_IR) usually, check lib convention. 
    # Actually rir-generator returns (nsample, n_mics) usually?
    # Let's check typical usage. If standard pypi rir-generator:
    # returns [samples, mics]
    
    if h.shape[0] != nsample: # If it's mic x samples
        h = h.T
        
    M = mics.shape[0]
    mic_signals = []
    
    # Convolve each channel
    for i in range(M):
        # convolve
        out = signal.fftconvolve(clean_signal, h[:, i], mode='full')
        mic_signals.append(out[:len(clean_signal)]) # Keep same length
        
    mic_signals = np.array(mic_signals).T # (Samples, Mics)
    
    # Add Noise
    # SNR = 10 log10 (P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR/10)
    
    sig_pow = np.mean(mic_signals**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_pow), mic_signals.shape)
    
    return mic_signals + noise

def compute_stft(signals):
    """Compute STFT of multi-channel signals."""
    # signals: (Time, Mics)
    # Return: (Freq, Time, Mics)
    stfts = []
    for i in range(signals.shape[1]):
        f, t, Zxx = signal.stft(signals[:, i], fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP_LEN)
        stfts.append(Zxx)
    
    # Stack to (Freq, Time, Mics)
    # Zxx is (Freqs, Time)
    return np.stack(stfts, axis=-1)

def get_steering_vector(freqs, look_pos, mic_pos):
    """
    Calculate steering vector for a specific point.
    freqs: (F,)
    look_pos: (3,)
    mic_pos: (M, 3)
    Returns: (F, M)
    """
    # Time delay: tau_i = distance(look, mic_i) / C
    # Phase shift: exp(-j * 2pi * f * tau)
    
    # Distances
    dists = np.linalg.norm(mic_pos - look_pos, axis=1) # (M,)
    # Relative to center or absolute? Standard is absolute delay for Phase alignment
    
    # However, usually we align to first mic or center.
    # Let's use absolute.
    taus = dists / C
    
    # (F, 1) * (1, M) -> (F, M)
    phases = -2j * np.pi * freqs[:, None] * taus[None, :]
    steering = np.exp(phases)
    
    # Normalize roughly? Typically steering vectors are unit magnitude per element (already is exp(jx)).
    # We might normalize by 1/sqrt(M) for beamforming conventions.
    return steering / np.sqrt(mic_pos.shape[0])

# --- SRP-PHAT ---
def srp_phat(stft_data, mic_pos, grid_x, grid_y, z_height=1.5):
    """
    SRP-PHAT implementation.
    stft_data: (F, T, M)
    Returns: Energy map (GridY, GridX)
    """
    F_bins = stft_data.shape[0]
    freqs = np.fft.rfftfreq(N_FFT, 1/FS)
    # Avoid DC
    freq_idx = range(1, F_bins) 
    
    # Compute Cross-Power Spectrum Phase (generalized cross correlation)
    # GCC-PHAT(i, j) = X_i * conj(X_j) / |X_i * conj(X_j)|
    # Sum over time frames to get robust estimate
    
    M = stft_data.shape[2]
    
    # Precompute global PHAT measures
    # Pairs (i, j)
    # This is expensive if we do it for all pairs for all grid points.
    # Efficient way: Beamforming approach.
    # Output Power P(r) = sum_f | w^H(f, r) * X(f) |^2
    # For PHAT, we whiten X: X_phat = X / |X|.
    # Then P(r) = sum_f | w^H * X_phat |^2
    
    # Normalized STFT
    mag = np.abs(stft_data)
    mag[mag < 1e-10] = 1e-10
    X_phat = stft_data / mag
    
    # Average over time frames (optional, or sum power later)
    # We can average the covariance matrix or the signal
    # SRP usually sums GCC over all pairs.
    # Which is equivalent to Delay-and-Sum beamforming on Whitened signals.
    
    # Let's iterate over grid
    energy_map = np.zeros((len(grid_y), len(grid_x)))
    
    # Precompute Steering Vectors is hard due to memory (Res*Res*Freq*M)
    # Loop over grid
    for iy, y in enumerate(grid_y):
        for ix, x in enumerate(grid_x):
            pos = np.array([x, y, z_height])
            sv = get_steering_vector(freqs, pos, mic_pos) # (F_all, M)
            
            # Select frequencies
            sv = sv[freq_idx, :]
            X_curr = X_phat[freq_idx, :, :] # (F, T, M)
            
            # Beamform: Y(f, t) = sv^H * X
            # (F, 1, M) dot (F, T, M) -> (F, T)
            # einsum: f m, f t m -> f t (conj sv)
            
            bf = np.einsum('fm,ftm->ft', sv.conj(), X_curr)
            
            # Power
            pow_val = np.sum(np.abs(bf)**2)
            energy_map[iy, ix] = pow_val
            
    return energy_map

# --- MUSIC ---
def music(stft_data, mic_pos, grid_x, grid_y, z_height=1.5):
    """
    Narrowband MUSIC aggregated over frequencies or BroadBand MUSIC.
    Simple approach: Incoherent Narrowband MUSIC. Sum pseudo-spectrum over freq bins.
    """
    F_bins = stft_data.shape[0]
    freqs = np.fft.rfftfreq(N_FFT, 1/FS)
    freq_idx = range(10, F_bins) # Skip low freqs, usually noisy or poor res
    
    M = stft_data.shape[2]
    
    # Covariance Matrices: R(f) = E[x(f) x(f)^H]
    # Average over Time
    # shape: (F, M, M)
    R = np.einsum('ftm,ftn->fmn', stft_data, stft_data.conj())
    R /= stft_data.shape[1] # Normalize by time frames
    
    # Subspace decomposition
    # Eigen decomposition of R(f)
    # Sort eigenvalues, noise subspace is associated with (M-1) smallest eigenvalues for 1 source.
    # Vectors: U_n
    
    noise_subspaces = []
    
    for f in freq_idx:
        vals, vecs = np.linalg.eigh(R[f])
        # Vals are sorted ascending.
        # Signal subspace: Largest eigenvalue (index -1)
        # Noise subspace: Remaining (0 to M-2)
        # U_n: (M, M-1)
        U_n = vecs[:, :-1]
        noise_subspaces.append(U_n)
        
    spectrum_map = np.zeros((len(grid_y), len(grid_x)))
    
    for iy, y in enumerate(grid_y):
        for ix, x in enumerate(grid_x):
            pos = np.array([x, y, z_height])
            sv = get_steering_vector(freqs, pos, mic_pos) # (F_all, M)
            
            denom_sum = 0
            
            for i, f in enumerate(freq_idx):
                a = sv[f, :] # (M,)
                U_n = noise_subspaces[i] # (M, M-1)
                
                # Projection: a^H * U_n * U_n^H * a
                proj = np.abs(np.vdot(a, U_n @ U_n.T.conj() @ a))
                denom_sum += proj
            
            # P_music = 1 / sum(proj)
            if denom_sum < 1e-9: denom_sum = 1e-9
            spectrum_map[iy, ix] = 1.0 / denom_sum
            
    return spectrum_map


# --- Runners ---

def get_peak_location(emap, grid_x, grid_y):
    """Find location of max value in map."""
    iy, ix = np.unravel_index(np.argmax(emap), emap.shape)
    return np.array([grid_x[ix], grid_y[iy]])

def run_q1(speech_file):
    print("Running Q1...")
    
    # Setup
    snr = 15
    t60 = 0.3
    
    # Random source
    sx = np.random.uniform(SOURCE_X_RANGE[0], SOURCE_X_RANGE[1])
    sy = np.random.uniform(SOURCE_Y_RANGE[0], SOURCE_Y_RANGE[1])
    source_pos = np.array([sx, sy, 1.5])
    
    print(f"True Source: {source_pos}")
    
    # Signal
    clean = load_audio(speech_file)
    signals = generate_signals(clean, source_pos, ROOM_DIMS, MICS, t60, snr)
    
    # STFT
    stft_data = compute_stft(signals)
    
    # SRP-PHAT
    srp_map = srp_phat(stft_data, MICS, X_GRID, Y_GRID)
    est_srp = get_peak_location(srp_map, X_GRID, Y_GRID)
    
    # MUSIC
    music_map = music(stft_data, MICS, X_GRID, Y_GRID)
    est_music = get_peak_location(music_map, X_GRID, Y_GRID)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(srp_map, origin='lower', extent=[X_GRID[0], X_GRID[-1], Y_GRID[0], Y_GRID[-1]], aspect='auto')
    plt.colorbar(label='Power')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.scatter(source_pos[0], source_pos[1], c='r', marker='x', label='True')
    plt.scatter(est_srp[0], est_srp[1], c='w', marker='o', edgecolors='k', label='Est')
    plt.title(f'SRP-PHAT (Err={np.linalg.norm(est_srp - source_pos[:2]):.2f}m)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.imshow(music_map, origin='lower', extent=[X_GRID[0], X_GRID[-1], Y_GRID[0], Y_GRID[-1]], aspect='auto')
    plt.colorbar(label='Spectrum')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.scatter(source_pos[0], source_pos[1], c='r', marker='x', label='True')
    plt.scatter(est_music[0], est_music[1], c='w', marker='o', edgecolors='k', label='Est')
    plt.title(f'MUSIC (Err={np.linalg.norm(est_music - source_pos[:2]):.2f}m)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('HW3/q1_maps.png')
    plt.close()
    
    # Text summary append
    with open("HW3/results.txt", "a") as f:
        f.write(f"\n--- Q1 Results ---\n")
        f.write(f"True Source: {source_pos}\n")
        f.write(f"SRP-PHAT Est: {est_srp}, Error: {np.linalg.norm(est_srp - source_pos[:2]):.4f}\n")
        f.write(f"MUSIC Est: {est_music}, Error: {np.linalg.norm(est_music - source_pos[:2]):.4f}\n")
        if np.array_equal(est_srp, est_music):
            f.write("(Note: Both algorithms peaked at the same grid point, hence identical error.)\n")


def run_q2(speech_file):
    print("Running Q2...")
    
    # Monte Carlo Parameters
    N_TRIALS = 30
    
    # Scenarios:
    # 1. Varying Noise (T60=300ms, snr=[5, 15, 30])
    # 2. Varying Rev (SNR=15dB, t60=[0.15, 0.30, 0.55])
    
    # Overlap: SNR=15, T60=300 is in both.
    
    snrs = [5, 15, 30]
    t60s = [0.15, 0.30, 0.55]
    
    # We need to collect errors for:
    # Fixed T60=0.3, vary SNR
    # Fixed SNR=15, vary T60
    
    results_snr = {'srp': [], 'music': []}  # Avg RMSE per SNR
    results_t60 = {'srp': [], 'music': []}  # Avg RMSE per T60
    
    # Let's run the specific scenarios required
    
    # Part 1: Vary SNR (fix T60=0.3)
    # Part 2: Vary T60 (fix SNR=15)
    
    # To be efficient, we can structure loops.
    # Scenarios list
    scenarios = []
    # (snr, t60, type)
    for s in snrs:
        scenarios.append({'snr': s, 't60': 0.3, 'group': 'snr_vary'})
    for t in t60s:
        if t == 0.3: continue # Already done in snr list (15, 0.3) if we want to reuse, but for simplicity let's just re-run or cache.
        # Actually question asks for 6 scenarios. SNR=15, T60=300 is the pivot.
        # Let's just run all combinations needed.
        scenarios.append({'snr': 15, 't60': t, 'group': 't60_vary'})
        
    # The set of unique (snr, t60) pairs:
    # (5, 0.3), (15, 0.3), (30, 0.3)
    # (15, 0.15), (15, 0.55)
    # Total 5 unique pairs. Usually the center one is shared.
    
    pairs = [
        (5, 0.3), (15, 0.3), (30, 0.3),
        (15, 0.15), (15, 0.55)
    ]
    
    # Store errors
    # Map pair -> list of errors
    errors_srp = {p: [] for p in pairs}
    errors_music = {p: [] for p in pairs}
    
    clean = load_audio(speech_file)
    
    for i in range(N_TRIALS):
        if i % 5 == 0: print(f"Trial {i}/{N_TRIALS}")
        
        # Random loc
        sx = np.random.uniform(SOURCE_X_RANGE[0], SOURCE_X_RANGE[1])
        sy = np.random.uniform(SOURCE_Y_RANGE[0], SOURCE_Y_RANGE[1])
        source_pos = np.array([sx, sy, 1.5])
        
        for (snr, t60) in pairs:
            # Generate
            signals = generate_signals(clean, source_pos, ROOM_DIMS, MICS, t60, snr)
            stft_data = compute_stft(signals)
            
            # SRP
            srp_map = srp_phat(stft_data, MICS, X_GRID, Y_GRID)
            est_srp = get_peak_location(srp_map, X_GRID, Y_GRID)
            err_srp = np.linalg.norm(est_srp - source_pos[:2])
            errors_srp[(snr, t60)].append(err_srp)
            
            # MUSIC
            music_map = music(stft_data, MICS, X_GRID, Y_GRID)
            est_music = get_peak_location(music_map, X_GRID, Y_GRID)
            err_music = np.linalg.norm(est_music - source_pos[:2])
            errors_music[(snr, t60)].append(err_music)
            
    # Compute RMSE
    rmse_srp = {}
    rmse_music = {}
    for p in pairs:
        rmse_srp[p] = np.sqrt(np.mean(np.array(errors_srp[p])**2))
        rmse_music[p] = np.sqrt(np.mean(np.array(errors_music[p])**2))

    # --- Plotting Results ---
    
    # 1. Error vs SNR (fixed T60=0.3)
    snr_x = [5, 15, 30]
    y_srp = [rmse_srp[(s, 0.3)] for s in snr_x]
    y_music = [rmse_music[(s, 0.3)] for s in snr_x]
    
    plt.figure()
    plt.plot(snr_x, y_srp, 'o-', label='SRP-PHAT')
    plt.plot(snr_x, y_music, 's-', label='MUSIC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (m)')
    plt.title('Localization Error vs Noise (T60=300ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('HW3/q2_snr.png')
    plt.close()
    
    # 2. Error vs T60 (fixed SNR=15)
    t60_x = [0.15, 0.30, 0.55]
    y_srp_t = [rmse_srp[(15, t)] for t in t60_x]
    y_music_t = [rmse_music[(15, t)] for t in t60_x]
    
    plt.figure()
    plt.plot(t60_x, y_srp_t, 'o-', label='SRP-PHAT')
    plt.plot(t60_x, y_music_t, 's-', label='MUSIC')
    plt.xlabel('T60 (s)')
    plt.ylabel('RMSE (m)')
    plt.title('Localization Error vs Reverberation (SNR=15dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('HW3/q2_t60.png')
    plt.close()
    
    # Write stats
    with open("HW3/results.txt", "a") as f:
        f.write(f"\n--- Q2 Results (RMSE over {N_TRIALS} trials) ---\n")
        f.write("Scenario (SNR, T60) | SRP-PHAT RMSE | MUSIC RMSE\n")
        f.write("-" * 50 + "\n")
        for p in pairs:
            f.write(f"{p}           | {rmse_srp[p]:.4f}        | {rmse_music[p]:.4f}\n")

if __name__ == "__main__":
    # Ensure output dir
    if not os.path.exists("HW3"):
        os.makedirs("HW3")
    
    # Clear previous results
    with open("HW3/results.txt", "w") as f:
        f.write("HW3 Simulation Results\n======================\n")
        
    # Use local file in HW3 folder
    audio_path = os.path.join("HW3", "speech.wav")
    
    if not os.path.exists(audio_path):
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
        else:
            print("Error: HW3/speech.wav not found. Please copy a wav file there or provide path.")
            sys.exit(1)
            
    print(f"Using audio: {audio_path}")
    run_q1(audio_path)
    run_q2(audio_path)
    print("Done. Results saved to HW3/results.txt and plots.")
