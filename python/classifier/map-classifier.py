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
from multibeam.utils import *
import numpy as np
import scipy
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

    # dump pings to disk
    pingu = ping*255.0
    fname = 'pings/raw/'+str(msg.time)
    cv2.imwrite(fname+'.png',pingu.astype(np.uint8))
    didson.saveConfig(fname+'.json')

    # TODO: iSNR can and should be re-estimated with every image

    didson.noise = 0.002
    # deconvolve
    ping_deconv = didson.deconvolve(ping)
    # remove beam pattern taper
    # ping_deconv = didson.removeTaper(ping_deconv)
    # remove range effects
    # ping_e3 = didson.removeRange(ping_e2)

    # intensity distribution estimation
    (p,k) = getMixtureParameters(ping_deconv)
    print ' pi1  |  pi2  |  s1   |  s2   |  kld   '
    print format(p[0],'0.3f'),'|',format(p[1],'0.3f'),'|',format(p[2],'0.3f'),'|',format(p[3],'0.3f'),'|',format(10000*k,'0.2f')
    # segmentation
    ping_binary = segment_map(ping_deconv, p[0],p[1], p[2], p[3])
    # TODO: condition on p[1] to discard segmentatio output
    # range extraction
    bin_length = (didson.max_range - didson.min_range)/(didson.num_bins + 0.0)
    ranges = []
    intensities = [] 

    if (any(p<0)):
        # invalid fit; skip!
        print '[WARN] Invalid fit; reverting to fixed threshold'
        # segmentation
        ping_binary = np.copy(ping_deconv);
        ping_binary[ping_binary<0.3] = 0
        # pings are (512,96)
        # range extraction
        intensities = np.amax(ping_binary,axis=0)
        ranges = np.argmax(ping_binary, axis=0)
        ranges = ranges*bin_length;
        ranges[ranges<=0] = -didson.min_range # no return here
        ranges += didson.min_range*np.ones(msg.width)
        ranges = -1*np.ones(96)
        intensities = np.zeros(96)
    elif (p[1]< 0.001):
        # no object present
        print '[WARN] No object present'
        ranges = -1*np.ones(96)
        intensities = np.zeros(96)
    else:
        (ranges,intensities) = extract_max(ping_deconv, ping_binary, didson.min_range,bin_length)

    # classify
    # ping_binary = ping_deconv;
    # ping_binary[ping_binary<0.3] = 0
    # # pings are (512,96)
    # # range extraction
    # intensities = np.amax(ping_binary,axis=0)
    # ranges = np.argmax(ping_binary, axis=0)
    # ranges = ranges*bin_length;
    # ranges[ranges<=0] = -didson.min_range # no return here
    # ranges += didson.min_range*np.ones(msg.width)

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

    # publish
    lcm_node.publish("MULTIBEAM_RANGES", msg_out.encode()) 

    print 'intensity range:', format( ping_deconv.min(),'0.2f' ), '-', format( ping_deconv.max(),'0.2f' )
    print ''

    # show results - disable for speed improvements
    ranges = np.argmax(ping_binary, axis=0)
    ping_hits = np.zeros_like(ping_binary)
    ping_hits[ranges,range(0,96)] = 1.0
    ping_deconv_cart = didson.toCart(ping_deconv)
    ping_hits = didson.toCart(ping_hits)
    img_raw = didson.toCart(ping)
    img_deconv = didson.toCart(ping_deconv)

    img_hits = np.dstack((img_raw,img_raw,img_raw))
    img_hits[:,:,2] =  img_hits[:,:,2] + ping_hits 

    cv2.imshow('ping (raw)',img_raw)
    cv2.imshow('ping (enhanced)',img_deconv)
    cv2.imshow('ping (hits)',img_hits)

    cv2.waitKey(1)


if __name__ == '__main__':
    print '[multibeam.classifier.main]'
    print '[feature/mrf]'
    print '[2018-03-08]'
    print 'NumPy:', np.__version__
    print 'SciPy:', scipy.__version__

    global lcm_node, didson

    # instantiate a sonar object with default config
    didson = Didson();
    didson.loadConfig('data/DIDSON/didson.json')

    lcm_node = lcm.LCM()
    ping_subscription = lcm_node.subscribe("MULTIBEAM_PING", pingHandler)

    # this breaks with the conda-installed version of opencv
    cv2.namedWindow('ping (raw)',cv2.WINDOW_NORMAL)
    cv2.namedWindow('ping (enhanced)',cv2.WINDOW_NORMAL)
    cv2.namedWindow('ping (hits)',cv2.WINDOW_NORMAL)

    try:
        while True:
            lcm_node.handle()
    except KeyboardInterrupt:
        pass

