#!/usr/bin/env python

"""
Multibeam range classification

This script subscribes to lcm messages of the type multibeam.ping_t on the
channel MULTIBEAM_PING. Each received ping is processed using the multibeam module
functionality (enhanced and classified), and the resulting range measurements area
published on the MULTIBEAM_RANGES channel
"""

from multibeam.sonar import Sonar
# from multibeam.didson import Didson
from multibeam.utils import *
import numpy as np
import cv2
import lcm
import json

import slam_pb2

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

def updateDidson(msg):
    global didson

    if ( didson.min_range != msg.min_range ) or (didson.max_range != msg.max_range):
        # window parameters have changed
        print 'window parameters have changed, resetting window'
        print 'min_range',didson.min_range,'->',msg.min_range
        print 'max_range',didson.max_range,'->',msg.max_range
        didson.reset_window(msg.min_range, msg.max_range)

    if (didson.rx_gain!=msg.rx_gain):
        didson.rx_gain = msg.rx_gain

    # TODO: update other config parameters (focus, gain, etc)

def save_ping(msg, ping_img):
    global didson
    scan = slam_pb2.Scan()

    scan.platform_pose.x = msg.platform_origin[0]
    scan.platform_pose.y = msg.platform_origin[1]
    scan.platform_pose.z = msg.platform_origin[2]
    scan.platform_pose.qw = msg.platform_orientation[0]
    scan.platform_pose.qx = msg.platform_orientation[1]
    scan.platform_pose.qy = msg.platform_orientation[2]
    scan.platform_pose.qz = msg.platform_orientation[3]

    scan.sensor_pose.x = msg.sensor_origin[0]
    scan.sensor_pose.y = msg.sensor_origin[1]
    scan.sensor_pose.z = msg.sensor_origin[2]
    scan.sensor_pose.qw = msg.sensor_orientation[0]
    scan.sensor_pose.qx = msg.sensor_orientation[1]
    scan.sensor_pose.qy = msg.sensor_orientation[2]
    scan.sensor_pose.qz = msg.sensor_orientation[3]

    fname = 'scan_'+str(msg.time)+'.bpb'
    with open(fname, 'w') as fp:
        fp.write(scan.SerializeToString())
        fp.close()


def savePing(msg, ping_img):
    global didson
    ping = {}
    ping['timestamp']= msg.time
    ping['id'] = msg.sonar_id

    ping['platform']={}
    ping['platform']['x'] = msg.platform_origin[0]
    ping['platform']['y'] = msg.platform_origin[1]
    ping['platform']['z'] = msg.platform_origin[2]
    ping['platform']['qw'] = msg.platform_orientation[0]
    ping['platform']['qx'] = msg.platform_orientation[1]
    ping['platform']['qy'] = msg.platform_orientation[2]
    ping['platform']['qz'] = msg.platform_orientation[3]

    ping['sensor']={}
    ping['sensor']['x'] = msg.sensor_origin[0]
    ping['sensor']['y'] = msg.sensor_origin[1]
    ping['sensor']['z'] = msg.sensor_origin[2]
    ping['sensor']['qw'] = msg.sensor_orientation[0]
    ping['sensor']['qx'] = msg.sensor_orientation[1]
    ping['sensor']['qy'] = msg.sensor_orientation[2]
    ping['sensor']['qz'] = msg.sensor_orientation[3]

    ping['hfov'] = msg.hfov
    ping['vfov'] = msg.vfov

    ping['num_beams'] = msg.num_beams
    ping['beam_hfov'] = msg.beam_hfov
    ping['beam_vfov'] = msg.beam_vfov

    ping['rx_gain'] =  msg.rx_gain

    ping['min_range'] = msg.min_range
    ping['max_range'] = msg.max_range
    ping['focus'] = msg.focus

    ping['num_bins'] = msg.num_bins

   # TODO: fix this! there is a non-linear look-up between beam and angle
    # ping['azimuths'] = np.linspace(msg.hfov/2.0, -msg.hfov/2.0, msg.num_beams).tolist()
    ping['azimuths'] = didson.azimuths.tolist()
    ping['ranges'] = np.linspace(msg.min_range, msg.max_range, msg.num_bins).tolist()

    ping['beams'] = {}
    for i in range(0,msg.num_beams):
        ping['beams'][str(i)] = np.squeeze( ping_img[:,i] ).tolist()

    ping['taper'] = didson.taper.tolist()
    ping['psf'] = np.squeeze( didson.psf ).tolist()
    ping['noise'] = didson.noise

    fname = str(msg.time)+'.json'
    with open(fname, 'w') as fp:
        json.dump(ping, fp, sort_keys=True, indent=2)


def pingHandler(channel, data):
    global lcm_node, didson
    msg = ping_t.decode(data)

    # check if we need to update the didson object
    updateDidson(msg)

    ping = np.copy(np.asarray(msg.image, dtype=np.int16))
    ping.shape = (msg.height, msg.width)

    # convert to [0,1] range 
    # TODO: replace hard-coded conversions with "astype"
    ping = ping.astype(np.float64)
    ping+=32768.0 # convert to range 0 - 65535
    ping/=65535.0 # convert to range 0 - 1

    # dump pings to disk
    pingu = ping*255.0
    fname = 'pings/'+str(msg.time)
    cv2.imwrite(fname+'_raw_polar.png',pingu.astype(np.uint8))
    didson.save_config(fname+'.json')

    pingc = didson.to_cart(ping, 255.0)
    pingcu = pingc*255.0
    cv2.imwrite(fname+'_raw_cart.png',pingcu.astype(np.uint8))

    # deconvolve
    ping_deconv = didson.deconvolve(ping)
    # remove beam pattern taper
    # ping_deconv = didson.removeTaper(ping_deconv)
    # remove range effects
    # ping_e3 = didson.removeRange(ping_e2)
    savePing(msg, ping)
    save_ping(msg, ping)

    # classify
    ping_binary = ping_deconv;
    ping_binary[ping_binary<0.3] = 0
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

    # publish
    lcm_node.publish("MULTIBEAM_RANGES", msg_out.encode())

    print 'intensity ranges:'
    print '   ping:', ping.min(), ping.max()
    print '   ping_deconv:', ping_deconv.min(), ping_deconv.max()

    # show results - disable for speed improvements
    ranges = np.argmax(ping_binary, axis=0)
    ping_hits = np.zeros_like(ping_binary)
    ping_hits[ranges,range(0,96)] = 1.0
    ping_deconv_cart = didson.to_cart(ping_deconv)
    ping_hits = didson.to_cart(ping_hits)
    img_raw = didson.to_cart(ping)
    img_deconv = didson.to_cart(ping_deconv)

    img_hits = np.dstack((img_raw,img_raw,img_raw))
    img_hits[:,:,2] =  img_hits[:,:,2] + ping_hits 

    cv2.imshow('ping (raw)',img_raw)
    cv2.imshow('ping (enhanced)',img_deconv)
    cv2.imshow('ping (hits)',img_hits)

    cv2.waitKey(1)


if __name__ == '__main__':
    print '[multibeam.classifier.main]'
    print '[branch: feature/mrf]'
    print '[date:   2019-04-30]'

    global lcm_node, didson

    # instantiate a sonar object with default config
    didson = Sonar();
    didson.load_config('data/DIDSON/didson.json')

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

