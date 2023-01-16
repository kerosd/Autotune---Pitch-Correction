#!/usr/bin/python3
from functools import partial
from pathlib import Path
import argparse #argparse kütüphanesi ile bir python programına gelen argümanları kontrol altına alabiliriz.
import librosa #Müzik ve ses analizi için bir python paketi.
import librosa.display
import numpy as np # Python programlama dili için büyük, çok boyutlu dizileri ve matrisleri destekleyen, bu diziler üzerinde çalışacak üst düzey matematiksel işlevler ekleyen bir kütüphanedir.
import matplotlib.pyplot as plt # Matplotlib veri görselleştirme için kullandığımız 2 boyutlu bir çizim kütüphanesidir.
import soundfile as sf # NumPy tabanlı bir ses kütüphanesidir.
import scipy.signal as sig #Scipy alt modülüdür, sinyal işleme için kullanılır.
import psola #Bu modül perde kaydırma işlemi için kullanılır.


SEMITONES_IN_OCTAVE = 12 # Semitone (yarım ton)12 tonluk bir ölçekte iki bitişik not arasındaki aralık olarak tanımlanır.


def degrees_from(scale: str):
    """Gam a karşılık gelen perdeler döndürülür."""
    degrees = librosa.key_to_degrees(scale)
    # Perde değiştirmeyi ölçekten en yakın dereceye düzgün bir şekilde gerçekleştirmek için, 
    # bir oktav yükseltilmiş birinci dereceyi tekrarlamamız gerekir. 
    # Aksi takdirde, temel dereceden biraz daha düşük perdeler yanlış atanır.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE])) #Concatenate string verileri birleştiren bir operatördür
    return degrees


def closest_pitch(f0):
    """Verilen perde değerleri en yakın MIDI nota numaralarına yuvarlanır"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values. (NaN matematik ve bilgisayar biliminde sayısal olmayan bir değeri tanımlamak için kullanılan bir terimdir)
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Midi notası Hz. oluyor.
    return librosa.midi_to_hz(midi_note)


def closest_pitch_from_scale(f0, scale):
    """Verilen ölçeğe ait f0'a en yakın perdeyi döndür"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Ölceğe en yakın perde bulunur.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Giriş perdesi sınıfı ile istenen perde sınıfı arasındaki farkı hesaplayın.
    degree_difference = degree - degrees[degree_id]
    # Giriş MIDI nota numarası hesaplanan fark kadar kaydırılır.
    midi_note -= degree_difference
    # Midi notası Hz. oluyor.
    return librosa.midi_to_hz(midi_note)


def aclosest_pitch_from_scale(f0, scale):
    """f0 dizisindeki her perdeyi verilen ölçeğe ait en yakın perdeyle eşleyin."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Düzeltilen perdeyi ek olarak yumuşatmak için medyan filtreleme gerçekleştirilir.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    # Perde smoothed bir şekilde döner.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, plot=False):
    # autotune işlemi için temel parametreler ayarlanır.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Bir nota veya ton kaydının perdesini veya temel frekansını tahmin etmek için, perde algılama algoritması kullanıyoruz (PYIN algorithm).
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    # correction uygulanır.
    corrected_f0 = correction_function(f0)

    if plot:
        # Orjinal perde yörüngesi ile ayarlanan perde yörüngesinin spektogramı çizilir.
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    # PSOLA algoritmasını kullanarak perde kaydırma.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument('vocals_file')
    ap.add_argument('--plot', '-p', action='store_true', default=False,
                    help='if set, will produce a plot of the results')
    ap.add_argument('--correction-method', '-c', choices=['closest', 'scale'], default='closest')
    ap.add_argument('--scale', '-s', type=str, help='see librosa.key_to_degrees;'
                                                    ' used only for the \"scale\" correction'
                                                    ' method')
    args = ap.parse_args()
    
    filepath = Path(args.vocals_file)

    # Ses dosyası yüklenir.
    y, sr = librosa.load(str("lil_yachty_vocal.wav"), sr=None, mono=False)

    # Sadece mono dosyalar işlenir. Eğer stereo ise yalnızca ilk kanal kullanılır.
    if y.ndim > 1:
        y = y[0, :]

    # Perde ayarlama stratejisi seçiliyor.
    correction_function = closest_pitch if args.correction_method == 'closest' else \
        partial(aclosest_pitch_from_scale, scale=args.scale)

    # auto-tuning uygulanıyor.
    pitch_corrected_y = autotune(y, sr, correction_function, args.plot)

    # "pitch correction" işlemi uygulanan ses çıktı dosyasına yazdırılıyor.
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)

    
if __name__=='__main__':
    main()
    
