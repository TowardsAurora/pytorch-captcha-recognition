import time


def start_server():
    server = "python -m visdom.server"
    osOrder = "__import__('os')"
    sy = ".system('"
    fil = "')"
    eval(osOrder + sy + server + fil)


if __name__ == '__main__':
    start_server()
