import pyaudio
import wave
from cnn_melspec import build_model, infer

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"
model_path = "trained/save.ptr"



def recorder():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("开始！")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("结束。")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def recognizer():
    model, __ = build_model(model_path)
    return infer(model, WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    print("孤立词语音识别系统")
    while True:
        print("开始录音？Y/N: ", end="")
        flag = input()
        if flag == "Y" or flag == "y":
            print("请在提示后清晰地念出词语，你只有2秒的时间")
            recorder()
            # result = recognizer()
            # print(result)
        else:
            print("录音结束")
            break
