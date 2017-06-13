#!/usr/bin/env python

"""
Multibeam range classification

This script subscribes to lcm messages of the type multibeam.ping_t on the
channel MULTIBEAM_PING. Each received ping is processed using the multibeam module
functionality (enhanced and classified), and the resulting range measurements area
published on the MULTIBEAM_RANGES channel
"""

from multibeam.sonar import Sonar
from multibeam.didson import Didson
from matplotlib import pyplot as plt
import numpy as np
import cv2
import lcm

# multibeam-lcmtypes
from multibeamlcm import *

__author__     = "Pedro Vaz Teixeira"
__copyright__  = ""
__credits__    = ["Pedro Vaz Teixeira"]
__license__    = ""
__version__    = "1.0.0"
__maintainer__ = "Pedro Vaz Teixeira"
__email__      = "pvt@mit.edu"
__status__     = "Development"


def pingHandler(channel, data):
    global lcm_node, didson 
    msg = ping_t.decode(data)

# check if we need to update the didson object

    if ( didson.min_range != msg.min_range ) or (didson.max_range != msg.max_range):
        # window parameters have changed
        print 'window parameters have changed, resetting window'
        print 'min_range',didson.min_range,'->',msg.min_range
        print 'max_range',didson.max_range,'->',msg.max_range
        didson.resetWindow(msg.min_range, msg.max_range)

    ping = np.copy(np.asarray(msg.image, dtype=np.int16))
    ping.shape = (msg.height, msg.width)

    # convert to [0,1] range 
    # TODO: replace hard-coded conversions with "astype"
    ping = ping.astype(np.float64)
    ping+=32768.0 # convert to range 0 - 65535
    ping/=65535.0 # convert to range 0 - 1

    # deconvolve
    ping_deconv = didson.deconvolve(ping)
    # remove beam pattern taper
    ping_deconv = didson.removeTaper(ping_deconv)
    # remove range effects

    print 'ping:', ping.min(), '-', ping.max()
    print 'ping_deconv:', ping_deconv.min(), '-', ping_deconv.max()

    bins = np.arange(0,1.0,0.01)
    h = np.histogram(ping, bins)
    hd = np.histogram(ping_deconv, bins)
    plt.clf()
    plt.plot(bins[:-1], (h[0]+0.0)/np.sum(h[0]))
    plt.plot(bins[:-1], (hd[0]+0.0)/np.sum(hd[0]))
    plt.xlim([0, 1.0])
    plt.yscale('log')
    plt.grid()
    plt.legend(['raw','enh'])
    plt.pause(0.001)

    # show results - disable for speed improvements
    # img = didson.toCart(ping)
    # cv2.imshow('ping (raw)',img)
    # img_deconv = didson.toCart(ping_deconv)
    # cv2.imshow('ping (enhanced)',img_deconv)
    # cv2.waitKey(1)

if __name__ == '__main__':    

    print '[multibeam.classifier.main]'

    global lcm_node, didson

    # instantiate a sonar object with default config
    didson = Didson();
    didson.loadConfig('data/DIDSON/didson.json')

    lcm_node = lcm.LCM()
    ping_subscription = lcm_node.subscribe("MULTIBEAM_PING", pingHandler)

    # this breaks with the conda-installed version of opencv
    cv2.namedWindow('ping (raw)',cv2.WINDOW_NORMAL)
    cv2.namedWindow('ping (enhanced)',cv2.WINDOW_NORMAL)
    
    try:
        while True:
            lcm_node.handle()
    except KeyboardInterrupt:
        pass

