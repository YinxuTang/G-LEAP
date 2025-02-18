from multiprocessing import Pool
import os
import time


def f(x):
    # print("module name:", __name__)
    print("module: {:s}, ppid: {:d}, pid: {:d}, BEGIN".format(__name__, os.getppid(), os.getpid()))
    time.sleep(2 * x)
    print("module: {:s}, ppid: {:d}, pid: {:d}, END".format(__name__, os.getppid(), os.getpid()))
    return x * x


if __name__ == '__main__':
    with Pool(5) as p:
        p.map(f, [1, 2, 3])
