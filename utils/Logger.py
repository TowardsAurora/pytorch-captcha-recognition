class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.txt', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.txt', "a")
        f.write(message + '\n')
        f.close()