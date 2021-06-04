"""
Variants of Accelerometers, Gyroscope and Magnetometers 
for the usage of vertex position data (surface simulation).
ModdedTrajectories needs to be used.
""" 
import numpy as np
from imusim.platforms.sensors import *
from imusim.environment.gravity import STANDARD_GRAVITY
from imusim.utilities.documentation import prepend_method_doc


""" Accelerometers """ 

class SurfaceAccelerometer(Sensor):
    """ Base class for accelerometers with surface sampled data. """

    
    def trueValues(self, t):
        """
        Returns a vector of values, one for each axis, that would
        be measured by an ideal sensor of this type.
        """

        gravity = self.platform.simulation.environment.gravitationalField
        g = gravity(self.trajectory.position(t), t)
        l = self.trajectory.acceleration(t)
        a = (l - g) 
        return a 


class IdealSurfaceAccelerometer(SurfaceAccelerometer, IdealSensor):
    """ An ideal accelerometer """
    pass 


class IdealGravitySurfaceSensor(IdealSurfaceAccelerometer):
    """
    An ideal fictional sensor that measures only acceleration due to gravity.
    """
    def trueValues(self, t):
        gravity = self.platform.simulation.environment.gravitationalField
        g = gravity(self.trajectory.position(t), t)
        return self.trajectory.rotation(t).rotateFrame(-g)


class NoisySurfaceAccelerometer(NoisySensor, IdealSurfaceAccelerometer):
    """
    Accelerometer with additive white Gaussian noise.
    """
    pass


class NoisyTransformedSurfaceAccelerometer(NoisyTransformedSensor, IdealSurfaceAccelerometer):
    """
    Accelerometer with affine transform transfer function and Gaussian noise.
    """
    pass



""" Gyroscopes """ 

class SurfaceGyroscope(Sensor):
    """ Base class for gyroscopes of surfaces """
    def trueValues(self, t):
        omega = self.trajectory.rotationalVelocity(t).vector
        return self.trajectory.rotation(t).rotateFrame(omega)   
        # TODO: check if this is correct with GlobalRotationTrajectory


class IdealSurfaceGyroscope(SurfaceGyroscope, IdealSensor):
    """ An Ideal gyroscope """
    pass 


class NoisySurfaceGyroscope(SurfaceGyroscope, NoisySensor):
    """
    Gyroscope with additive white Gaussian noise.
    """
    pass


class NoisyTransformedGyroscope(NoisySurfaceGyroscope, NoisyTransformedSensor):
    """
    Gyroscope with affine transform transfer function and Gaussian noise.
    """
    pass



""" Magnetometer """ 

class SurfaceMagnetometer(Sensor):
    """ Base class for magnetometers """

    def trueValues (self, t):
        return self.platform.simulation.environment.magneticField(self.trajectory.position(t), t)


class IdealSurfaceMagnetometer(SurfaceMagnetometer, IdealSensor):
    """
    An ideal magnetometer.
    """
    pass 


class NoisySurfaceMagnetometer(NoisySensor, IdealSurfaceMagnetometer):
    """
    Magnetometer with additive white Gaussian noise.
    """
    pass


class NoisyTransformedSurfaceMagnetometer(NoisyTransformedSensor, IdealSurfaceMagnetometer):
    """
    Magnetometer with an affine transform transfer function and Gaussian noise.
    """
    pass

