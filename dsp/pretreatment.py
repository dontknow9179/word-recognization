import os
import re
import multiprocessing
import shutil

def is_target(name):
    if re.match("^[0-9]{11}-[0-9]{2}-[0-9]{2}\.wav$", name) is not None:
        return True
    return False


def dfs_dir(path):
    curr = os.listdir(path)
    # print(curr)
    files = []
    for item in curr:
        npath = os.path.join(path, item)
        if os.path.isdir(npath):
            files += dfs_dir(npath)
        elif os.path.isfile(npath):
            if is_target(item):
                files.append(npath)
    return files


def move_file(src):
    name = os.path.basename(src)
    dst_dir = os.path.join("E:/speech_data", name[:11])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst = os.path.join(dst_dir, name)
    shutil.copy(src, dst)
    print("moving {0} to {1}".format(src, dst))


def move_files(srcs):
    for src in srcs:
        move_file(src)


def move():
    files = dfs_dir("E:/speech_rawdata")
    print("{0} file to be moved".format(len(files)))
    pnum = 30
    tnum = len(files) // pnum + 1
    sub_task = [files[i:i + tnum] for i in range(0, len(files), tnum)]
    # print(sum([len(t) for t in sub_task]))
    for i in range(pnum):
        p = multiprocessing.Process(target=move_files, args=(sub_task[i],))
        p.start()


def check():
    base = "./speech_data"
    for item in os.listdir(base):
        files = os.listdir(os.path.join(base, item))
        file_num = len(files)
        if file_num != 400:
            print("{0} has {1} files".format(item, file_num))
            for i in range(0, 20):
                for j in range(1, 21):
                    name = "{0}-{1:02}-{2:02}.wav".format(item, i, j)
                    if name not in files:
                        print(name, " lost")


if __name__ == "__main__":
    # move()
    check()
