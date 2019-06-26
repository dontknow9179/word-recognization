import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
# import warnings
import matplotlib.pyplot as plt
# warnings.filterwarnings("error")

padto = 55000


def mel_spec(path, count_retry=0):
    # 将有问题的音频文件随机替换成同一个人的同一个词的另一个音频
    if count_retry > 5:
        raise IOError("retry>5 {}".format(path))
    try:
        y, sr = librosa.load(path)
    except:
        file = os.path.basename(path)
        path = os.path.dirname(path)
        file = file[:15] + "{:02}".format(np.random.randint(1, 21)) + file[17:]
        return mel_spec(os.path.join(path, file), count_retry + 1)
    
    # padding填充
    l = len(y)
    print(l)
    left_pad = (padto - l) // 2
    right_pad = padto - left_pad - l
    y = np.pad(y, (left_pad, right_pad), 'wrap')

    # 分帧，加窗，短时傅里叶变换
    feat = librosa.stft(y, hop_length=256, n_fft=512)
    # 绝对值再平方得到能量谱
    feat = np.abs(feat) ** 2

    # 梅尔频谱
    try:
        melspec = librosa.feature.melspectrogram(S=feat, sr=sr, n_mels=128)
    except:
        raise IOError("melspectrpgram error {}".format(path))

    # 取对数
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    return log_melspec
    
def specshow(feat):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feat, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_melspec = mel_spec("../speech_data/16307130343/16307130343-00-01.wav")
    specshow(log_melspec)
