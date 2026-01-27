
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
SOURCE_Z = 1.5

# Grid for Heatmaps (GRID_RESxGRID_RES)
GRID_RES = 20
X_GRID = np.linspace(SOURCE_X_RANGE[0], SOURCE_X_RANGE[1], GRID_RES)
Y_GRID = np.linspace(SOURCE_Y_RANGE[0], SOURCE_Y_RANGE[1], GRID_RES)

# FFT parameters
N_FFT = 256
HOP_LEN = 128

# Number of speaker locations for Question 2
Q2_N_LOCATIONS = 30


# --- Helper Functions ---

def load_audio(filepath, target_len_sec=2.0):
    """Load and trim/pad audio to target length."""
    rate, data = wavfile.read(filepath)
    
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
    nsample = int(t60 * FS) if t60 > 0 else 4096 
    
    h = rir.generate(
        c=C,
        fs=FS,
        r=mics,
        s=source_pos,
        L=room_dims,
        reverberation_time=t60,
        nsample=nsample
    )
    
    if h.shape[0] != nsample: 
        h = h.T
        
    M = mics.shape[0]
    mic_signals = []
    
    # Convolve each channel
    for i in range(M):
        out = signal.fftconvolve(clean_signal, h[:, i], mode='same')
        mic_signals.append(out[:len(clean_signal)]) 
        
    mic_signals = np.array(mic_signals).T 
    
    # Add Noise
    # Robust Signal Power: Active Speech Only
    # Threshold: 1% of max amplitude (approx -40dB)
    threshold = 0.01 * np.max(np.abs(mic_signals))
    active_mask = np.abs(mic_signals) > threshold
    
    # If we have enough active samples, use them. Else fall back to mean.
    if np.sum(active_mask) > 100:
        sig_pow = np.mean(mic_signals[active_mask]**2)
    else:
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
    dists = np.linalg.norm(mic_pos - look_pos, axis=1) 
    taus = dists / C
    
    phases = -2j * np.pi * freqs[:, None] * taus[None, :]
    steering = np.exp(phases)
    
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
    # Band-limit to 300-3000 Hz
    freq_idx = np.where((freqs >= 300) & (freqs <= 3000))[0] 
    
    # Compute Cross-Power Spectrum Phase (generalized cross correlation)
    
    M = stft_data.shape[2]
    
    # Normalized STFT for PHAT
    mag = np.abs(stft_data)
    mag[mag < 1e-10] = 1e-10
    X_phat = stft_data / mag
    
    # SRP usually sums GCC over all pairs (equivalent to Delay-and-Sum on Whitened signals).
    
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
    # Band-limit to 300-3000 Hz
    freq_idx = np.where((freqs >= 300) & (freqs <= 3000))[0]
    
    M = stft_data.shape[2]
    
    # Covariance Matrices: R(f) = E[x(f) x(f)^H]
    R = np.einsum('ftm,ftn->fmn', stft_data, stft_data.conj())
    R /= stft_data.shape[1] 
    
    # Subspace decomposition: Eigen decomposition of R(f)
    # Noise subspace is associated with (M-1) smallest eigenvalues for 1 source.
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

def run_q1(speech_file, source_pos):
    print("Running Q1...")
    
    # Setup
    snr = 15
    t60 = 0.3
    
    # Use provided source_pos
    # source_pos = np.array([sx, sy, 1.5])
    
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
    # Save plot to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "q1_maps.png"))
    plt.close()
    
    # Text summary append
    results_path = os.path.join(script_dir, "results.txt")
    with open(results_path, "a") as f:
        f.write(f"\n--- Q1 Results ---\n")
        f.write(f"True Source: {source_pos}\n")
        f.write(f"SRP-PHAT Est: {est_srp}, Error: {np.linalg.norm(est_srp - source_pos[:2]):.4f}\n")
        f.write(f"MUSIC Est: {est_music}, Error: {np.linalg.norm(est_music - source_pos[:2]):.4f}\n")
        if np.array_equal(est_srp, est_music):
            f.write("(Note: Both algorithms peaked at the same grid point, hence identical error.)\n")


def run_q2(speech_file, source_locations):
    print("Running Q2...")
    
    # Monte Carlo Parameters
    N_TRIALS = len(source_locations)
    
    pairs = [
        (5, 0.3), (15, 0.3), (30, 0.3),
        (15, 0.15), (15, 0.55)
    ]
    
    # Store errors
    # Map pair -> list of errors
    errors_srp = {p: [] for p in pairs}
    errors_music = {p: [] for p in pairs}
    
    clean = load_audio(speech_file)
    
    for i, source_pos in enumerate(source_locations):
        if i % 5 == 0: print(f"Trial {i}/{N_TRIALS}")
               
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
    # Determine script dir locally if needed, or pass it. 
    # For simplicity, calculate again or use robust path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "q2_snr.png"))
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
    plt.savefig(os.path.join(script_dir, "q2_t60.png"))
    plt.close()
    
    # Write stats
    results_path = os.path.join(script_dir, "results.txt")
    with open(results_path, "a") as f:
        f.write(f"\n--- Q2 Results (RMSE over {N_TRIALS} trials) ---\n")
        f.write("Scenario (SNR, T60) | SRP-PHAT RMSE | MUSIC RMSE\n")
        f.write("-" * 50 + "\n")
        for p in pairs:
            f.write(f"{p}           | {rmse_srp[p]:.4f}        | {rmse_music[p]:.4f}\n")

if __name__ == "__main__":
    # Determine script directory for robust path handling
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define output file paths
    results_path = os.path.join(SCRIPT_DIR, "results.txt")
    
    # Clear previous results
    with open(results_path, "w") as f:
        f.write("HW3 Simulation Results\n======================\n")
        
    # Use local file in HW3 folder
    audio_path = os.path.join(SCRIPT_DIR, "speech.wav")
    
    if not os.path.exists(audio_path):
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
        else:
            print(f"Error: {audio_path} not found. Please ensure speech.wav is in the same directory as the script.")
            sys.exit(1)
            
            
    print(f"Using audio: {audio_path}")
    
    # Generate random locations for consistent testing
    source_locations = []
    for _ in range(Q2_N_LOCATIONS):
        sx = np.random.uniform(SOURCE_X_RANGE[0], SOURCE_X_RANGE[1])
        sy = np.random.uniform(SOURCE_Y_RANGE[0], SOURCE_Y_RANGE[1])
        source_locations.append(np.array([sx, sy, 1.5]))
        
    # Run Q1 with the FIRST location from the set
    run_q1(audio_path, source_locations[0])
    
    # Run Q2 with the ALL locations
    run_q2(audio_path, source_locations)
    print(f"Done. Results saved to {results_path} and plots.")
