#!/usr/bin/env python

"""
Multibeam range classification

This script subscribes to lcm messages of the type multibeam.ping_t on the
channel MULTIBEAM_PING. Each received ping is processed using the multibeam module
functionality (enhanced and classified), and the resulting range measurements area
published on the MULTIBEAM_RANGES channel. Additionally, this script can also save
the ping information as a combination of Protobuf, JSON, and PNG.
"""

from multibeam.sonar import Sonar
from multibeam.utils import *
import numpy as np
import cv2
import lcm
import json

import slam_pb2 # proto

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
    """
    update sonar configuration
    """
    global didson

    if (didson.min_range != msg.min_range) or (didson.max_range != msg.max_range):
        print 'window parameters have changed, resetting window'
        print '[', didson.min_range, ',', didson.max_range, '] -> [', msg.min_range, ',', msg.max_range, ']'
        didson.reset_window(msg.min_range, msg.max_range, 0.02)

    if didson.rx_gain != msg.rx_gain:
        didson.rx_gain = msg.rx_gain

    # TODO: update other config parameters (focus, gain, etc)

def save_ping(msg, ping_img, ranges, intensities):
    """
    Save the scan as a serialized protobuffer object
    """

    global didson
    scan = slam_pb2.Scan()

    scan.timestamp = msg.time

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

    scan.num_beams = msg.num_beams
    scan.min_range = msg.min_range
    scan.max_range = msg.max_range
    scan.hfov = msg.hfov
    scan.vfov = msg.vfov

    for i in range(0, msg.num_beams):
        scan.ranges.append(ranges[i])

        scan.intensities.append(intensities[i]) # return intensity

        azi = -didson.azimuth(i) # flip sign
        # azi = msg.hfov/2.0 - i*msg.beam_hfov # WORKS

        beam = scan.beams.add()
        beam.x = 0.0
        beam.y = 0.0
        beam.z = 0.0
        beam.qw = np.cos(azi/2.0)
        beam.qx = 0.0
        beam.qy = 0.0
        beam.qz = np.sin(azi/2.0)

    fname = 'scan_'+str(msg.time)+'.bpb'
    with open(fname, 'w') as fp:
        fp.write(scan.SerializeToString())
        fp.close()


def savePing(msg, ping_img):
    global didson
    ping = didson.to_json(ping_img)

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

    ping['beam_hfov'] = msg.beam_hfov
    ping['beam_vfov'] = msg.beam_vfov

    ping['focus'] = msg.focus

    ping['ranges'] = np.linspace(msg.min_range, msg.max_range, msg.num_bins).tolist()

    fname = str(msg.time)+'.json'
    with open(fname, 'w') as fp:
        json.dump(ping, fp, sort_keys=True, indent=2)


def pingHandler(channel, data):
    global lcm_node, didson
    msg = ping_t.decode(data)

    updateDidson(msg) # check if we need to update the didson object

    ping = np.copy(np.asarray(msg.image, dtype=np.int16))
    ping.shape = (msg.height, msg.width)

    # convert to [0,1] range 
    # TODO: replace hard-coded conversions with "astype"
    ping = ping.astype(np.float64)
    ping+=32768.0 # convert to range 0 - 65535
    ping/=65535.0 # convert to range 0 - 1

    fname = 'pings/'+str(msg.time)
    cv2.imwrite(fname+'_raw_polar.png',(ping*255.0).astype(np.uint8))
    didson.save_config(fname+'.json')

    ping_deconv = didson.preprocess(ping, False)

    bin_length = (didson.max_range - didson.min_range)/(didson.num_bins + 0.0)
    pulse = get_template(dr=bin_length)
    idx = segment_smap(ping_deconv, pulse)
    idx[idx < 1] = -1
    ranges = np.copy(idx).astype(float)
    ranges[ranges > 0] *= bin_length
    ranges[ranges > 0] += didson.min_range*np.ones_like(ranges[ranges>0])
    intensities = ping_deconv[idx, range(0, msg.num_beams)]
    intensities[idx < 1] = 0

    save_ping(msg, ping, ranges, intensities) # save as proto

    # ping_binary = np.copy(ping_deconv);
    # ping_binary[ping_binary<0.3] = 0
    # pings are (512,96)
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

    for i in range(0, msg.num_beams):
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

    # show results - disable for speed improvements
    ping_hits = reconstruct(ping, idx)
    ping_hits = didson.to_cart(ping_hits)

    img_raw = didson.to_cart(ping)
    img_deconv = didson.to_cart(ping_deconv)

    img_hits = np.dstack((img_raw,img_raw,img_raw))
    img_hits[:,:,2] =  img_hits[:,:,2] + ping_hits 

    cv2.imshow('ping (raw)',np.rot90(img_raw,3))
    cv2.imshow('ping (enhanced)',np.rot90(img_deconv,3))
    cv2.imshow('ping (hits)',np.rot90(img_hits,3))

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

