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

    # classify
    # ideally just a call to getReturns
    ping_binary = ping_deconv;
    ping_binary[ping_binary<0.35] = 0
    # pings are (512,96)
    intensities = np.amax(ping_binary,axis=0)
    ranges = np.argmax(ping_binary, axis=0)
    bin_length = (didson.max_range - didson.min_range)/(didson.num_bins + 0.0)
    ranges = ranges*bin_length;
    ranges[ranges<=0] = -didson.min_range # no return here
    ranges += didson.min_range*np.ones(msg.width)

    msg_out = range_scan_t()
    msg_out.time = msg.time
    msg_out.platform_origin = msg.platform_origin
    msg_out.platform_orientation = msg.platform_orientation
    msg_out.sensor_origin = msg.sensor_origin
    msg_out.sensor_orientation = msg.sensor_orientation
    msg_out.num_beams = msg.num_beams

    for i in range(0,msg.num_beams):
        # create and fill range_t message
        b = range_t()
        b.time = msg.time
        b.beam_origin = np.zeros(3)
        # TODO: get value from message
        angle = msg.hfov/2.0 - i*msg.beam_hfov
        qw = np.cos(angle/2.0)
        qx = qy = 0.0
        qz = np.sin(angle/2.0)
        # TODO: renormalize quaternion!
        b.beam_orientation = np.array([qw, qx, qy, qz]) 
        b.range = ranges[i]
        b.strength = intensities[i]
        b.hfov = msg.beam_hfov
        b.vfov = msg.beam_vfov
        b.min_range = msg.min_range
        b.max_range = msg.max_range
        b.has_intensity = False

        # add it to the range_scan_t message
        msg_out.beams.append(b)

    # create outgoing message
    # msg_out = planar_lidar_t() 
    # msg_out.utime = msg.time
    # msg_out.nranges = didson.num_beams
    # msg_out.ranges = ranges #np.zeros(didson.num_beams)
    # msg_out.nintensities = didson.num_beams
    # msg_out.intensities = intensities #np.zeros(didson.num_beams)
    # msg_out.rad0 = didson.fov/2.0 
    # msg_out.radstep = -didson.fov/(didson.num_beams + 0.0)

    # publish
    lcm_node.publish("MULTIBEAM_RANGES", msg_out.encode()) 

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

