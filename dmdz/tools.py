import datetime
import numpy as np

def print_array(x, precision=6, suppress=True):
    """
    Pretty prints a numpy array to the given precision.
    """
    with np.printoptions(precision=precision, suppress=suppress):
        print(x)


def timestamp(time_format='%Y%m%dT%H%M%S'):
    """
    Timestamp. Defaults to YYYYmmDDTHHMMSS format, where the character T separates
    date from time. This is a condensed version of ISO formatting.
    :return: Timestamp string.
    """
    now = datetime.datetime.now()
    return now.strftime(time_format)


def timeprint(string=None, delimiter=": "):
    """
    Print timestamp followed by optional string predended by delimiter.
    e.g. timestamp_print('foo') prints "YYYYmmDDTHHMMSS: foo"
    :param string:
    :param delimiter: Delimiter between timestamp and print message. By default ": ". Note that it includes the space character.
    """
    if string:
        string = delimiter+string
    print("%s%s" % (timestamp(),string))