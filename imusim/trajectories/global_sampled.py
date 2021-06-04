
from imusim.maths.quaternions import Quaternion, QuaternionArray, QuaternionFromEuler
from imusim.utilities.caching import CacheLastValue
from imusim.utilities.time_series import TimeSeries
import imusim.maths.vectors as vectors
import numpy as np
from imusim.trajectories.sampled import SampledRotationTrajectory
from imusim.trajectories.base import PositionTrajectory, RotationTrajectory
from imusim.trajectories.splined import * 
from bisect import bisect_left


"""
This files contains modded SampledTrajectories for the IMUSim environment in order to avoid using splines.
The acceleration and rotations are calculated directly if not already given as argument.
The positions are already in the global end position, so no rotation needs to be applied.
"""

class GlobalSampledPositionTrajectory(PositionTrajectory):
    """
    Represents a sampled position trajectory.

    @ivar positionKeyFrames: A L{TimeSeries} of position key frames.
    """

    def __init__(self, keyFrames=None):
        """
        Initialise trajectory.

        @param keyFrames: A L{TimeSeries} of position key frame samples.
        """
        self.positionKeyFrames = TimeSeries() if keyFrames is None else keyFrames
        self.timestamps = self.positionKeyFrames.timestamps.tolist()
        self.values = self.positionKeyFrames.values.tolist()
        assert(len(self.timestamps) == len(self.values[0]))


    def getIndex(self, t):
        """
        Returns the index of nearest given time t.
        """
        pos = bisect_left(self.timestamps, t)

        if pos == 0:
            return pos
        if pos == len(self.timestamps):
            return pos - 1

        before_val = self.timestamps[pos - 1]
        curr_val = self.timestamps[pos]

        if curr_val - t < t - before_val:
            return pos
        else:
            return pos - 1

    def getPositionByIndex(self, index):
        if len(self.positionKeyFrames) == 0:
            if np.ndim(t) == 0:
                return vectors.nan()
            else:
                return vectors.nan(len(t))
        else:
            x = self.values[0][index]
            y = self.values[1][index]
            z = self.values[2][index]
            arr = np.array([x,y,z])
            return arr.reshape((3,1))

    @CacheLastValue()
    def position(self, t):
        if len(self.positionKeyFrames) == 0:
            if np.ndim(t) == 0:
                return vectors.nan()
            else:
                return vectors.nan(len(t))
        else:
            return self.positionKeyFrames(t)


    @CacheLastValue()
    def velocity(self, t1):
        """
        Calculated by:  ((pos_i) - pos_i-1) / ((t_i) - t_i-1)
        """
        
        if t1 <= self.startTime:
            return 0
        elif t1 > self.endTime:
            return 0
        
        this_t = self.getIndex(t1)
        last_t = self.getIndex(t1) - 1 
        t0 = self.timestamps[last_t]

        return  (self.getPositionByIndex(this_t) - self.getPositionByIndex(last_t)) / (t1 - t0)


    @CacheLastValue()
    def acceleration(self, t1):
        """
        Calculated by:  (pos_i+1 + pos_i-1 - 2*pos_i) / (time_interval*time_interval)
        """
        if t1 <= self.startTime:
            return 0
        elif t1 >= self.endTime:
            return 0
        
        # get the last and next time step 
        last_t = self.getIndex(t1) - 1
        this_t = self.getIndex(t1)
        next_t = self.getIndex(t1) + 1
        
        t0 = self.positionKeyFrames.timestamps[last_t]
        t2 = self.positionKeyFrames.timestamps[next_t]        
        dt = t1 - t0
        return (self.getPositionByIndex(next_t) + self.getPositionByIndex(last_t) - 2*self.getPositionByIndex(this_t)) / (dt * dt)



    @property
    def startTime(self):
        return self.positionKeyFrames.earliestTime

    @property
    def endTime(self):
        return self.positionKeyFrames.latestTime

class GlobalSampledRotationTrajectory(RotationTrajectory):
    """
    Represents a sampled rotation trajectory.
    
    @ivar rotationKeyFrames: A L{TimeSeries} of rotiaton key frames.
    The TimeSeries consists of rotational matrices. One rotation matrix per sample. 
    """

    def __init__(self, keyFrames=None):
        """
        Initialise trajectory.

        @param keyFrames: A L{TimeSeries} of rotation key frame samples.
        """
        self.rotationKeyFrames = TimeSeries() if keyFrames is None else keyFrames

    @CacheLastValue()
    def rotation(self, t):
        if len(self.rotationKeyFrames) == 0:
            if np.ndim(t) == 0:
                return Quaternion.nan()
            else:
                return QuaternionArray.nan(len(t))
        else:
            return self.rotationKeyFrames(t)


    def getRotationByIndex(self, index):
        if len(self.rotationKeyFrames) == 0:
            if np.ndim(t) == 0:
                return Quaternion.nan()
            else:
                return QuaternionArray.nan(len(t))
        else:
            return self.rotationKeyFrames.values[index]


    def rotationalVelocity(self, t1):
        """
        r = rotation matrix
        omega = first derivative 
        alpha = second derivative

        r, omega, alpha = self._rotationSpline(t)   # automatically creates derivatives at specific timeframes
        return r.rotateVector(omega)
        ω(t) = 2θ∗(t) ⊗ θ′(t) 
        θ∗ (quaternion conjugate), ⊗ (quaternion multiplication)
        """
        if t1 <= self.startTime:
            return 0
        elif t1 > self.endTime:
            return 0
        
        
        last_t = self.getIndex(t1) - 1 
        this_t = self.getIndex(t1)
        t0 = self.timestamps[last_t]
        
        # retrieve quaternions 
        quat_t0 = self.getRotationByIndex(last_t)
        quat_t1 = self.getRotationByIndex(this_t)

        #normalise
        quat_t0 = quat_t0.normalise()
        quat_t1 = quat_t1.normalise()

        # order of euler angles: zyx
        euler0 = quat_t0.toEuler()
        euler1 = quat_t1.toEuler()

        dt = t1 - t0
        velocity = (euler1 - euler0) / dt
        
        # see quat_splines line 145 which returns to splined.py line 123
        # only the imaginary units of the acceleration quaternion is taken.
        velocity = QuaternionFromEuler(velocity, order='zyx', inDegrees=True)
        
        # analog to line 129 in file splined.py 
        return velocity
                
        
    def rotationalAcceleration(self, t1):
        """
        α(t) = 2θ∗(t) ⊗ θ′′(t)
        θ∗ (quaternion conjugate), ⊗ (quaternion multiplication)
        """
        if t1 <= self.startTime:
            return 0
        elif t1 >= self.endTime:
            return 0

        # get the last and next time step 
        last_t = self.getIndex(t1) - 1
        slast_t = self.getIndex(t1) - 2
        this_t = self.getIndex(t1)

        t0 = self.timestamps[last_t]
        t2 = self.timestamps[slast_t]

        # retrieve quaternions 
        quat_t0 = self.getRotationByIndex(slast_t)
        quat_t1 = self.getRotationByIndex(last_t)
        quat_t2 = self.getRotationByIndex(this_t)

        # normalize
        quat_t0 = quat_t0.normalise()
        quat_t1 = quat_t1.normalise()
        quat_t2 = quat_t2.normalise()

        euler0 = quat_t0.toEuler()
        euler1 = quat_t1.toEuler()
        euler2 = quat_t2.toEuler()

        dt = t1 - t0

        # imusim calculates first difference using velocities 
        acc = (euler2 + euler0 - 2*euler1) / (dt*dt)

        # see quat_splines line 145 which returns to splined.py line 123
        # only the imaginary units of the acceleration quaternion is taken.
        acc = QuaternionFromEuler(acc, order='zyx', inDegrees=True)

        # analog to line 129 in file splined.py 
        return acc


    @property
    def startTime(self):
        return self.rotationKeyFrames.earliestTime

    @property
    def endTime(self):
        return self.rotationKeyFrames.latestTime

class GlobalSampledTrajectory(GlobalSampledPositionTrajectory, GlobalSampledRotationTrajectory):
    """
    Represents a sampled position and rotation trajectory.
    """

    def __init__(self, positionKeyFrames=None, rotationKeyFrames=None, **kwargs):
        """
        Initialise trajectory.

        @param positionKeyFrames: L{TimeSeries} of position key frames.
        @param rotationKeyFrames: L{TimeSeries} of rotation key frames.
        """
        GlobalSampledPositionTrajectory.__init__(self, positionKeyFrames)
        GlobalSampledRotationTrajectory.__init__(self, rotationKeyFrames)
        

    @staticmethod
    def fromArrays(time, position, rotation):
        """
        Construct a L{SampledTrajectory} from time, position and rotation arrays.

        @param time: sequence of sample times.
        @param position: sequence of position samples.
        @param rotation: sequence of rotation samples.
        """
        return GlobalSampledTrajectory(
                TimeSeries(time, position),
                TimeSeries(time, rotation))

    @property
    def startTime(self):
        return max(self.positionKeyFrames.earliestTime, self.rotationKeyFrames.earliestTime) 

    @property
    def endTime(self):
        return min(self.positionKeyFrames.latestTime, self.rotationKeyFrames.latestTime) 
