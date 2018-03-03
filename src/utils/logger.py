"""
Simple logger class
"""

import time
import json
import numpy as np
import datetime as dt
import pandas as pd
import random


class arguments(object):
    def __init__(self, name):
        self.name = name


class loggerClass(object):
    def __init__(self, args, timestamp):
        self.events = []
        self.args = args
        self.ts = timestamp
        self._create_log()

    def _create_log(self):
        self.filename = '{}_{}'.format(self.args.name, self.ts)
        with open('../logs/{}.txt'.format(self.filename), 'w') as f:
            f.write('*' * 50)
            f.write('\n')
            f.write('LOGFILE: {} at {}'.format(self.args.name, self.ts))
            f.write('\n')
            f.write('*' * 50)
            f.write('\nargs: {} \n'.format(self.args))

    def log(self, event):
        with open('../logs/{}.txt'.format(self.filename), 'a') as f:
            f.write(event)
            f.write('\n')
        print(event)
