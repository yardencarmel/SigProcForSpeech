
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import rir_generator as rir
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility, ScaleInvariantSignalDistortionRatio
from scipy.signal import fftconvolve

import shutil
import urllib.request
from pathlib import Path

# --- 1. Constants & Configuration ---
BASE_DIR = Path(__file__).resolve().parent
ROOM_DIMS = [4, 5, 3]  # meters
FS = 16000
MICS_NUM = 5
MIC_DIST = 0.05  # meters
MIC_CENTER = [2, 1, 1.7]
SOURCE_POS_RTF = [2 + 1.5 * np.cos(np.deg2rad(30)), 1 + 1.5 * np.sin(np.deg2rad(30)), 1.7] # 30 deg, 1.5m
INTERFERENCE_POS = [2 + 2 * np.cos(np.deg2rad(150)), 1 + 2 * np.sin(np.deg2rad(150)), 1.7] # 150 deg, 2m
T60_VALUES = [0.15, 0.3] # seconds
SNR_VALUES = [0, 10] # dB
OUTPUT_DIR = BASE_DIR / 'output'
DATA_DIR = BASE_DIR / 'data'
MODEL_PATH = BASE_DIR / "dns64-a7761ff99a7d5bb6.th"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

# --- 2. Helper Functions ---

def generate_rir(t60, room_dims, mic_pos, source_pos, fs=FS):
    c = 340
    # Sabine formula approximation or just input t60 to rir_generator
    nsample = int(t60 * fs) if t60 > 0 else 4096
    
    # rir_generator expects numpy arrays
    h = rir.generate(
        c=c,
        fs=fs,
        r=mic_pos,
        s=source_pos,
        L=room_dims,
        reverberation_time=t60,
        nsample=nsample,
    )
    return h

def get_mic_array_pos(center, n_mics, dist, axis='x'):
    # center is [x, y, z]
    pos = np.zeros((n_mics, 3))
    for i in range(n_mics):
        offset = (i - (n_mics - 1) / 2) * dist
        if axis == 'x':
            pos[i] = [center[0] + offset, center[1], center[2]]
        elif axis == 'y':
            pos[i] = [center[0], center[1] + offset, center[2]]
    return pos

import soundfile as sf

def create_signals(speech_path, rir_impulse):
    # speech_path: path to .flac
    # rir_impulse: (n_mics, n_samples_rir)
    # Use soundfile to load
    audio, sample_rate = sf.read(speech_path)
    waveform = torch.from_numpy(audio.T if audio.ndim > 1 else audio).float()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        
    if sample_rate != FS:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=FS)
        waveform = resampler(waveform)
    
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    signal = waveform.numpy()[0]
    
    n_mics = rir_impulse.shape[1]
    signals = []
    for i in range(n_mics):
        s = fftconvolve(signal, rir_impulse[:, i], mode='full')
        signals.append(s)
    
    signals = np.array(signals) # (n_mics, len)
    return signals, signal

def add_white_noise(signal, snr_db):
    # Reference for SNR is usually the first mic signal component
    ref_pow = np.mean(signal[0] ** 2)
    noise_pow = ref_pow / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_pow), signal.shape)
    return signal + noise, noise

def add_interference_noise(signal, interference_rir, interference_source, snr_db):
    n_mics = interference_rir.shape[1]
    int_signals = []
    target_len = signal.shape[1]
    
    # Create interference signal at mics
    for i in range(n_mics):
         s_int = fftconvolve(interference_source, interference_rir[:, i], mode='full')
         int_signals.append(s_int)
    
    int_signals = np.array(int_signals)
    
    # Handle Length mismatch
    # If interference is too short, loop it? Or pad.
    if int_signals.shape[1] < target_len:
        # Repeat
        repeats = int(np.ceil(target_len / int_signals.shape[1]))
        int_signals = np.tile(int_signals, (1, repeats))
        
    if int_signals.shape[1] > target_len:
        int_signals = int_signals[:, :target_len]
        
    # Scale to target SNR based on Mic 0
    ref_sig_pow = np.mean(signal[0] ** 2)
    ref_int_pow = np.mean(int_signals[0] ** 2)
    
    if ref_int_pow > 0:
        alpha = np.sqrt(ref_sig_pow / (ref_int_pow * (10 ** (snr_db / 10))))
    else:
        alpha = 0
        
    scaled_interference = int_signals * alpha
    
    return signal + scaled_interference, scaled_interference

def compute_metrics(clean, est, fs=FS):
    if isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean)
    if isinstance(est, np.ndarray):
        est = torch.from_numpy(est)
        
    if clean.ndim == 1: clean = clean.unsqueeze(0)
    if est.ndim == 1: est = est.unsqueeze(0)
    
    # Mismatch length handling
    min_len = min(clean.shape[1], est.shape[1])
    clean = clean[:, :min_len]
    est = est[:, :min_len]

    pesq = PerceptualEvaluationSpeechQuality(fs, 'wb')
    estoi = ShortTimeObjectiveIntelligibility(fs)
    si_sdr = ScaleInvariantSignalDistortionRatio()
    
    try:
        p = pesq(est, clean).item()
    except:
        p = 0.0 # Handle very short duration or silence
        
    try:
        e = estoi(est, clean).item()
    except:
        e = 0.0

    try:
        s = si_sdr(est, clean).item()
    except:
        s = -100.0

    return {
        'PESQ': p,
        'ESTOI': e,
        'SI-SDR': s
    }

# --- 3. Beamformers ---

def delay_and_sum(signals, rir_ref):
    # Determine delay from RIR peaks
    peaks = np.argmax(np.abs(rir_ref), axis=0)
    ref_ch = MICS_NUM // 2 # Center mic
    delays = peaks - peaks[ref_ch] # Positive means this ch lags behind ref
    
    aligned_signals = []
    
    for i in range(MICS_NUM):
        d = delays[i]
        sig = signals[i]
        
        # Align signal by shifting contrary to delay
        s_aligned = np.roll(sig, -d)
        
        # Zero padding to handle wrap-around from roll
        if d > 0:
            s_aligned[-d:] = 0
        elif d < 0:
            s_aligned[:-d] = 0
            
        aligned_signals.append(s_aligned)
        
    ds_out = np.mean(aligned_signals, axis=0)
    return ds_out

def stft(x, n_fft=512, hop_length=256):
    return torch.stft(torch.from_numpy(x), n_fft=n_fft, hop_length=hop_length, return_complex=True, window=torch.hann_window(n_fft))

def istft(X, n_fft=512, hop_length=256):
    return torch.istft(X, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft)).numpy()

def mvdr_beamformer(noisy_signals, noise_only_signals):
    # Ensure inputs are float32
    if isinstance(noisy_signals, np.ndarray):
        noisy_signals = noisy_signals.astype(np.float32)
    if isinstance(noise_only_signals, np.ndarray):
        noise_only_signals = noise_only_signals.astype(np.float32)
        
    Y = stft(noisy_signals)
    N = stft(noise_only_signals)
    
    n_mics, n_freq, n_frames = Y.shape
    
    # Noise Covariance
    N = N.permute(1, 0, 2)
    # Average over time
    Phi_NN = torch.matmul(N, N.conj().transpose(1, 2)) / n_frames # (F, M, M)
    
    # Noisy Covariance (for GEVD RTF)
    Y_perm = Y.permute(1, 0, 2)
    Phi_YY = torch.matmul(Y_perm, Y_perm.conj().transpose(1, 2)) / n_frames
    
    d = torch.zeros(n_freq, n_mics, dtype=torch.cfloat)
    ref_mic = 2 # Center
    
    for f in range(n_freq):
        try:
             Pn = Phi_NN[f] + 1e-6 * torch.eye(n_mics)
             Py = Phi_YY[f]
             
             # GEVD: Phi_YY v = lambda Phi_NN v
             mat = torch.linalg.solve(Pn, Py)
             
             w, v = torch.linalg.eig(mat)
             idx = torch.argmax(w.real)
             eigenvec = v[:, idx]
             
             rtf_vec = eigenvec / (eigenvec[ref_mic] + 1e-10)
             d[f] = rtf_vec
        except:
             d[f] = torch.ones(n_mics)
             
    # MVDR Weights: w = (Phi_NN^-1 d) / (d^H Phi_NN^-1 d)
    w_mvdr = torch.zeros(n_freq, n_mics, dtype=torch.cfloat)
    
    for f in range(n_freq):
        Pn_inv = torch.linalg.inv(Phi_NN[f] + 1e-6 * torch.eye(n_mics))
        df = d[f].unsqueeze(1)
        
        num = torch.matmul(Pn_inv, df)
        denom = torch.matmul(df.conj().T, num)
        
        ws = num / (denom + 1e-10)
        w_mvdr[f] = ws.squeeze()
        
    # Apply
    Z = torch.zeros(n_freq, n_frames, dtype=torch.cfloat)
    for f in range(n_freq):
        wf = w_mvdr[f].unsqueeze(0)
        yf = Y_perm[f]
        Z[f] = torch.matmul(wf.conj(), yf)
        
    output = istft(Z)
    return output

# --- 4. Deep Learning ---
def load_model(name="dns64"):
    # Rely on installed denoiser package
    try:
        from denoiser.demucs import Demucs
    except ImportError:
        print("Error: 'denoiser' package not found. Please install it (e.g. pip install denoiser)")
        sys.exit(1)
    
    url = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"
    if not MODEL_PATH.exists():
        print(f"Downloading model {name}...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        
    # dna64 uses hidden=64, sample_rate=16000
    model = Demucs(hidden=64, sample_rate=16000)
    
    pkg = torch.load(MODEL_PATH, map_location='cpu')
    
    # Check if it's a package or direct state dict
    if isinstance(pkg, dict) and 'model' in pkg:
         model.load_state_dict(pkg['model'])
    else:
         # Assume raw state dict
         model.load_state_dict(pkg)
         
    model.eval()
    return model

def enhance_dl(model, noisy_signal):
    if model is None: return noisy_signal
    
    # Ensure tensor
    if isinstance(noisy_signal, np.ndarray):
        noisy_signal = torch.from_numpy(noisy_signal)
        
    noisy_signal = noisy_signal.float()
    
    x = noisy_signal
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0) 
    
    with torch.no_grad():
        out = model(x)
        
    return out.squeeze().numpy()

# --- Main ---
def main():
    # Expect data to be present in DATA_DIR (copied from HW1)
    files = list(DATA_DIR.glob('*.flac'))
    if not files:
        print(f"No speech files found in {DATA_DIR}! Please copy files from HW1.")
        sys.exit(1)
        
    print(f"Found {len(files)} files.")

    mic_pos = get_mic_array_pos(MIC_CENTER, MICS_NUM, MIC_DIST)
    source_pos = np.array(SOURCE_POS_RTF)
    interf_pos = np.array(INTERFERENCE_POS)
    
    results = [] 
    
    print("Loading DL Model...")
    dl_model = load_model()
    
    print("Starting processing...")
    
    # Limit to 20 files
    proc_files = files[:20]
    
    for t60 in T60_VALUES:
        print(f"Generating RIRs for T60={t60}...")
        rir_speech = generate_rir(t60, ROOM_DIMS, mic_pos, source_pos)
        rir_interf = generate_rir(t60, ROOM_DIMS, mic_pos, interf_pos)
        
        for snr in SNR_VALUES:
            for noise_type in ['gaussian', 'interference']:
                
                metrics_accum = {'PESQ': [], 'ESTOI': [], 'SI-SDR': []}
                dl_metrics_accum = {'PESQ': [], 'ESTOI': [], 'SI-SDR': []}
                
                print(f"Processing condition: T60={t60}, SNR={snr}, Noise={noise_type}")
                
                for f_idx, file_path in enumerate(proc_files):
                    # 1. Gen Signals
                    # Ensure file_path is string for sf.read if needed, or pass Path (sf supports Path usually)
                    mics_sig, src_sig = create_signals(str(file_path), rir_speech)
                    
                    # 2. Noise
                    if noise_type == 'gaussian':
                        noisy_mics, noise_only = add_white_noise(mics_sig, snr)
                    else:
                        int_file = files[(f_idx + 1) % len(files)]
                        
                        # Load using soundfile
                        i_audio, i_sr = sf.read(int_file)
                        int_wave = torch.from_numpy(i_audio.T if i_audio.ndim > 1 else i_audio).float()
                        if int_wave.ndim == 1: int_wave = int_wave.unsqueeze(0)
                        
                        if i_sr != FS:
                             resampler = torchaudio.transforms.Resample(orig_freq=i_sr, new_freq=FS)
                             int_wave = resampler(int_wave)
                             
                        int_src = int_wave.mean(0).numpy()
                        noisy_mics, noise_only = add_interference_noise(mics_sig, rir_interf, int_src, snr)

                    clean_center = mics_sig[2]
                    clean_first = mics_sig[0]
                    
                    # Q1: Plots/Wav (First condition only)
                    if f_idx == 0 and t60 == 0.3 and snr == 10 and noise_type == 'gaussian':
                         # Save
                         sf.write(OUTPUT_DIR / "q1_clean_mic1.wav", clean_first, FS)
                         sf.write(OUTPUT_DIR / "q1_noisy_mic1_gaussian.wav", noisy_mics[0], FS)
                         
                         plt.figure(figsize=(10, 8))
                         plt.subplot(3,1,1)
                         plt.title("Original Source")
                         plt.plot(src_sig) 
                         
                         plt.subplot(3,1,2)
                         plt.title("Clean Mic 1")
                         plt.plot(clean_first)
                         
                         plt.subplot(3,1,3)
                         plt.title("Noisy Mic 1")
                         plt.plot(noisy_mics[0])
                         plt.tight_layout()
                         plt.savefig(OUTPUT_DIR / "q1_plot_time.png")
                         plt.close()
                         
                         plt.figure(figsize=(10, 8))
                         plt.subplot(3,1,1)
                         plt.title("Spec Original")
                         plt.specgram(src_sig, Fs=FS)
                         plt.subplot(3,1,2)
                         plt.title("Spec Clean Mic 1")
                         plt.specgram(clean_first, Fs=FS)
                         plt.subplot(3,1,3)
                         plt.title("Spec Noisy Mic 1")
                         plt.specgram(noisy_mics[0], Fs=FS)
                         plt.tight_layout()
                         plt.savefig(OUTPUT_DIR / "q1_plot_freq.png")
                         plt.close()

                    # Processors
                    ds_out = delay_and_sum(noisy_mics, rir_speech)
                    mvdr_out = mvdr_beamformer(noisy_mics, noise_only)
                    
                    m = compute_metrics(clean_center, mvdr_out)
                    for k in m: metrics_accum[k].append(m[k])
                    
                    if f_idx == 0 and t60 == 0.3 and snr == 10:
                        # Save MVDR output for inspection
                        sf.write(OUTPUT_DIR / f"q2_mvdr_out_{noise_type}.wav", mvdr_out, FS)
                        
                    # DL
                    dl_out = enhance_dl(dl_model, noisy_mics[0])
                    m_dl = compute_metrics(clean_first, dl_out)
                    for k in m_dl: dl_metrics_accum[k].append(m_dl[k])
                    
                    if f_idx == 0 and t60 == 0.3 and snr == 10:
                        sf.write(OUTPUT_DIR / f"q3_dl_out_{noise_type}.wav", dl_out, FS)

                # Summarize
                res_row = {'T60': t60, 'SNR': snr, 'Noise': noise_type}
                for k in metrics_accum:
                    res_row[f"MVDR_{k}"] = np.mean(metrics_accum[k])
                for k in dl_metrics_accum:
                    res_row[f"DL_{k}"] = np.mean(dl_metrics_accum[k])
                results.append(res_row)

    # Write Report
    with open("results_report.md", "w") as f:
        f.write("# HW2 Results Report\n\n")
        f.write("| Reverberation (T60) [s] | SNR [dB] | Noise Type | MVDR PESQ | MVDR ESTOI | MVDR SI-SDR | DL PESQ | DL ESTOI | DL SI-SDR |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            line = f"| {r['T60']} | {r['SNR']} | {r['Noise']} | {r['MVDR_PESQ']:.3f} | {r['MVDR_ESTOI']:.3f} | {r['MVDR_SI-SDR']:.3f} | {r['DL_PESQ']:.3f} | {r['DL_ESTOI']:.3f} | {r['DL_SI-SDR']:.3f} |\n"
            f.write(line)
            
    print("Completion. Results saved to results_report.md")

if __name__ == "__main__":
    main()
