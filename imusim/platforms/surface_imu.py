"""
All alternative IMU variants which do not make use of
OffsetTrajecteries, since they are dedicated for RigidBody Data
e.g. Vertex measurements of SMPL Models.
"""

from imusim.platforms.imus import StandardIMU
from imusim.platforms import * 
from imusim.platforms.surface_sensor import IdealSurfaceAccelerometer, IdealSurfaceMagnetometer, IdealGravitySurfaceSensor, IdealSurfaceGyroscope
from imusim.platforms.adcs import IdealADC
from imusim.platforms.timers import IdealTimer
from imusim.platforms.radios import IdealRadio

"""Ideal IMU for rigid body sampled data on surfaces
    in order to avoid doubled rotation calculations,
    since the rotation of the offset is already in the
    given trajectory output.
    Only for the usage of GlobalSampledTrajectory """
class IdealSurfaceIMU(StandardIMU):
    """
    An IMU with idealised models for all components.
    """

    def __init__(self, simulation=None, trajectory=None):
        self.accelerometer = IdealSurfaceAccelerometer(self)
        self.magnetometer = IdealSurfaceMagnetometer(self)
        self.gyroscope = IdealSurfaceGyroscope(self)
        self.adc = IdealADC(self)
        self.timer = IdealTimer(self)
        self.radio = IdealRadio(self)
        StandardIMU.__init__(self, simulation, trajectory)


class MagicSurfaceIMU(StandardIMU):
    """
    An IMU with idealised components including a fictional gravity sensor.
    """
    def __init__(self, simulation=None, trajectory=None):
        self.accelerometer = IdealGravitySurfaceSensor(self)
        self.magnetometer = IdealSurfaceMagnetometer(self)
        self.gyroscope = IdealSurfaceGyroscope(self)
        self.adc = IdealADC(self)
        self.timer = IdealTimer(self)
        self.radio = IdealRadio(self)
        StandardIMU.__init__(self, simulation, trajectory)

