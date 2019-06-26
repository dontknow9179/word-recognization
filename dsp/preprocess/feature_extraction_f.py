import array
import warnings
import copy
import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import os
from preprocess.dataset import DataFeed
from matplotlib import pyplot as plt

from wave import open as open_wave


def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.
    ys: wave array
    amp: max amplitude (pos or neg) in result
    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    # return amp * ys / abs(np.max(ys) - np.min(ys))
    return amp * ys / max(high, low)


class Wave:

    def __init__(self, ys, ts=None, framerate=None):
        """Initializes the wave.
        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        self.ys = np.asanyarray(ys)
        if framerate is not None:
            self.framerate = framerate
        else:
            # self.framerate = 11025
            self.framerate = 22050
        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)


    def copy(self):
        """Makes a copy.
        Returns: new Wave
        """
        return copy.deepcopy(self)


    def __len__(self):
        return len(self.ys)


    def get_short_time_energy(self, seg_length):
        window = np.hamming(seg_length)
        i, j = 0, seg_length
        step = seg_length // 2

        # map from time to Spectrum
        ste = []
        x = []

        while j < len(self.ys):
            segment = self.ys[i:j]
            segment = segment * window

            # the nominal time for this segment is the midpoint
            t = (self.ts[i] + self.ts[j]) / 2
            x.append(t)
            ste.append(np.sum(segment * segment))

            i += step
            j += step

        return Wave(ste, ts=x, framerate=1 / (x[1] - x[0]))


    def get_short_time_cross_rate(self, seg_length):
        window = np.hamming(seg_length)
        i, j = 0, seg_length
        step = seg_length // 2
        # map from time to Spectrum
        stc = []
        x = []

        while j < len(self.ys):
            segment = self.ys[i:j]
            segment = segment * window

            cross_rate = np.sum(np.abs(np.diff(np.sign(segment)))) / 2

            # the nominal time for this segment is the midpoint
            t = (self.ts[i] + self.ts[j]) / 2
            x.append(t)
            stc.append(cross_rate)

            i += step
            j += step

        return Wave(stc, ts=x, framerate=1 / (x[1] - x[0]))


    def plot_short_time_feature(self, seg_length):
            ste = self.get_short_time_energy(seg_length)
            stc = self.get_short_time_cross_rate(seg_length)
            fig, (w, e, c) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            w.plot(self.ts, self.ys)
            w.set_title(u"波形")
            e.plot(ste.ts, ste.ys)
            e.set_title(u"短时能量")
            c.plot(stc.ts, stc.ys)
            c.set_title(u"短时平均过零率")
            # fig.show()
            plt.show()

    def endian_detection(self, plot=False, w_axe=None, e_axe=None, c_axe=None):
        frame_length = 0.05
        seg_lenght = int(frame_length * self.framerate) + 1
        ste = self.get_short_time_energy(seg_lenght)
        stc = self.get_short_time_cross_rate(seg_lenght)
        e_t = 0.98
        c_t = 120

        def search(es, cs):
            length = len(es)
            p = length
            for i in range(length):
                if es[i] > e_t:
                    p = i
                    break
            p_ = p
            for i in range(p):
                if cs[i] > c_t:
                    p_ = i
                    break
            return p_

        b = search(ste.ys, stc.ys)
        e = search(ste.ys[::-1], stc.ys[::-1]) # 逆序输入
        b, e = ste.ts[b], ste.ts[::-1][e]

        def search_index(ts, p):
            for i in range(len(ts)):
                if ts[i] > p:
                    return i
            return -1

        begin = search_index(self.ts, b)
        end = search_index(self.ts, e)

        if plot:
            fig, (w_axe, e_axe, c_axe) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            w_axe.plot(self.ts, self.ys)
            w_axe.set_title(u"波形")
            w_axe.plot(self.ts, self.ys, 'r')
            e_axe.plot(ste.ts, ste.ys)
            e_axe.set_title(u"短时能量")
            c_axe.plot(stc.ts, stc.ys)
            c_axe.set_title(u"短时平均过零率")
            plt.show()


    def pre_emphasis(self, alpha=0.97):
        self.ys[1:] = self.ys[1:] - alpha * self.ys[:-1]


    def extract_mel_feature(self, frame_size=512, overlap=0.5, nmel=64, nceps=12, normalize=True):
        self.pre_emphasis()
        wave_ys, wave_framerate = self.ys, self.framerate
        # 分帧
        samples = len(wave_ys)
        step_size = int(round(frame_size * overlap))
        n_frames = samples // step_size
        frames = np.hstack(
            [wave_ys[i: i + frame_size].reshape(-1, 1) for i in range(0, samples - frame_size, step_size)])
        # 一列为一帧的二维数组
        print(frames.shape, step_size, n_frames)

        # 加hamming窗
        hamming_window = np.hamming(frame_size)
        frames = frames * hamming_window.reshape(-1, 1)

        # 离散傅里叶变换 + 能量谱
        frames = np.abs(np.fft.rfft(frames)) ** 2

        # 加mel滤波器
        max_mel_freq = (2595 * np.log10(1 + wave_framerate / 2 / 700))
        min_mel_freq = 0
        mel_freqs = np.linspace(min_mel_freq, max_mel_freq, nmel + 2)
        hz_freqs = (700 * (10 ** (mel_freqs / 2595) - 1))
        # fft_freqs = np.fft.rfftfreq(frame_size, 1 / wave_framerate)
        freq_idx = np.floor((frame_size + 1) / wave_framerate * hz_freqs).astype(int)
        mel_filters = np.zeros([nmel, frame_size])
        for i in range(1, nmel + 1):
            l = freq_idx[i - 1]
            m = freq_idx[i]
            r = freq_idx[i + 1]
            for k in range(l, m):
                mel_filters[i - 1][k] = (k - l) / (m - l)
            for k in range(m, r):
                mel_filters[i - 1][k] = (r - k) / (r - m)

        melfreqfeat = mel_filters.dot(frames)
        melfreqfeat = 20 * np.log10(melfreqfeat)
        return melfreqfeat


def play_wave(filename='sound.wav'):
    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    # cmd = 'powershell -c (New-Object Media.SoundPlayer "{0}").PlaySync();'.format(filename)
    # cmd = 'start "{0}"'.format(filename)
    # print(cmd)
    # popen = subprocess.Popen(cmd, shell=True)
    # popen.communicate()
    import winsound
    winsound.PlaySound(filename, winsound.SND_FILENAME)


def read_wave(filename='sound.wav'):
    fp = open_wave(filename, 'r')

    nchannels = fp.getnchannels()
    nframes = fp.getnframes()
    sampwidth = fp.getsampwidth()
    framerate = fp.getframerate()

    z_str = fp.readframes(nframes)

    fp.close()

    dtype_map = {1: np.int8, 2: np.int16, 3: 'special', 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' % sampwidth)

    if sampwidth == 3:
        xs = np.fromstring(z_str, dtype=np.int8).astype(np.int32)
        ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
    else:
        ys = np.fromstring(z_str, dtype=dtype_map[sampwidth])

    # if it's in stereo, just pull out the first channel
    if nchannels == 2:
        ys = ys[::2]

    # ts = np.arange(len(ys)) / framerate
    wave = Wave(ys, framerate=framerate)
    wave.ys = normalize(wave.ys)
    return wave


def plot_wave(filename):
    wave = read_wave(filename)
    plt.plot(wave.ts, wave.ys)
    print(wave.framerate)
    plt.show()

def mel_spec(filename):
    wave = read_wave(filename)
    melfreqfeat = wave.extract_mel_feature
    return melfreqfeat

# def plot_spectrogram(filename):
#     wave = read_wave(filename)
#     spec = wave.make_spectrogram(1024)
#     spec.plot()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = DataFeed()
    # a, __ = data.get_by_id(3201)
    # w = read_wave(a)
    w = read_wave("speech_data/15307130233/15307130233-00-01.wav")
    w.pre_emphasis()
    plt.plot(w.ts, w.ys)
    plt.show()
    # play_wave(a)
    # import time

    # start = time.time()
    # print(start)
    # a, b = w.mfcc()
    # print(time.time() - start)
    # plt.imshow(a.T)
    # plt.show()
    # print(w.framerate)

    # w.endian_detection(plot=True)
    # plot_spectrogram(w)
    # np.random.seed(0)
    # for i in range(10):
    #     a, __ = data.get_by_id(np.random.randint(0, 32 * 20 * 20))
    #     wave = read_wave(a)
    #     wave.endian_detection(plot=True)
    # play_wav(a)
