""" Geometry of the device to create """


from shapely.lib import length
import tdgl 
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt


length_units = "um" # ummmmmm

def createShapes(bridge_width = 0.3, 
                 bulge_radius = 0.7, noise_amplitude = 0.02, noise_w = 100.0, circle_def = 0.01,
                 theta_min = np.pi / 2.0, theta_max = 0.0):
    
    width = bridge_width + bulge_radius
    length = 2*bulge_radius
    pExt = length / 10.0 

    fp = [[0, 0], (-pExt*2, 0), [-pExt*2, width], [length+pExt*2, width], [length+pExt*2, 0], [length, 0]]
   

    if theta_max // circle_def != 0:
        for t in np.linspace(0, theta_max, int(theta_max // circle_def)):
            fp.append(
                        [bulge_radius * np.cos(t) + bulge_radius, bulge_radius * np.sin(t)]
                    )

    for t in np.linspace(theta_max, theta_min, int((theta_min - theta_max) // circle_def)):
        fp.append([
            (bulge_radius + noise_amplitude * np.sin(noise_w * t)) * np.cos(t) + bulge_radius,
            (bulge_radius + noise_amplitude * np.sin(noise_w * t)) * np.sin(t)])


    for t in np.linspace(theta_min, np.pi, int((np.pi - theta_min) // circle_def)):
        fp.append([bulge_radius * np.cos(t) + bulge_radius, bulge_radius * np.sin(t)])
   
    film_points = np.array([np.array(p) for p in fp])

    left_term_points = [ [-pExt, 0], [-pExt*2, 0], [-pExt*2, width], [-pExt, width], [-pExt, 0] ]
    left_term = np.array([ np.array(p) for p in left_term_points])

    right_term_points = [ [length+pExt, 0], [length+2*pExt, 0], [length+2*pExt, width], [length+pExt, width], [length+pExt, 0] ]
    right_term = np.array([ np.array(p) for p in right_term_points])

    film = Polygon(film_points)

    probes = [np.array([ 0.0, width / 1.5 ]), np.array([length, width / 1.5 ])]
    
    
    return film, tdgl.Polygon("source", points = left_term), tdgl.Polygon("drain", points = right_term), probes
   

def createLayer(xi, london_lambda, d):
    return tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d, gamma=1)


def createDevice(layer,
                 bridge_width = 0.3, 
                 bulge_radius = 0.7, noise_amplitude = 0.02, noise_w = 100.0, circle_def = 0.01,
                 theta_min = np.pi / 2.0, theta_max = 0.0):
    """ Returns the already-made pyTDGL device with the geometry specified """

    out, source, drain, probe_points = createShapes(bridge_width, 
                                                    bulge_radius, noise_amplitude, noise_w, circle_def,
                                                    theta_min, theta_max
                                                    )

    film = (
        tdgl.Polygon("film", points=out)
        .resample(401)
        .buffer(0)
    )

    # probe_points = [probe_points[0], np.array([7, -3.5])]

    device = tdgl.Device(
        "weird_thing",
        layer=layer,
        film=film,
        terminals=[source, drain],
        probe_points=probe_points,
        length_units=length_units,
    )

    return device

