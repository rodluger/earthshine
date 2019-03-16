import spiceypy as spice
import glob
from tqdm import tqdm
import numpy as np
import starry
assert starry.__version__ == "1.0.0.dev0", \
    "This code requires the `starry` version 1.0.0.dev0."

# Constants
REARTH = 1.0 / 6371.0
TJD0 = 2457000

__all__ = ["design_matrix", "tess_vector"]


def tess_earth_vector(time):
    """Return the cartesian position of TESS relative to the Earth."""
    # Load the SPICE data
    ephemFiles = glob.glob('../data/TESS_EPH_PRE_LONG_2018*.bsp')
    tlsFile = '../data/tess2018338154046-41240_naif0012.tls'
    solarSysFile = '../data/tess2018338154429-41241_de430.bsp'
    for ephFil in ephemFiles:
        spice.furnsh(ephFil)
    spice.furnsh(tlsFile)
    spice.furnsh(solarSysFile)

    # JD time range
    allTJD = time + TJD0
    nT = len(allTJD)
    allET = np.zeros((nT,), dtype=np.float)
    for i, t in enumerate(allTJD):
        allET[i] = spice.unitim(t, 'JDTDB', 'ET')

    # Calculate positions of TESS, the Earth, and the Sun
    # Note that our reference frame is that of `starry`,
    # where `y` points *north*.
    # This is just the rotation {x, y, z} --> {z, x, y}
    # relative to the J200 mean equatorial coordinates.
    tess = np.zeros((3, len(allET)))
    for i, et in enumerate(allET):
        outTuple = spice.spkezr('Mgs Simulation', et, 'J2000', 'NONE', 'Earth')
        tess[0, i] = outTuple[0][1] * REARTH
        tess[1, i] = outTuple[0][2] * REARTH
        tess[2, i] = outTuple[0][0] * REARTH
    return tess


def design_matrix(time, ydeg=10, nt=2, period=1.0, phase0=0.0, 
                  fit_linear_term=False):
    """
    Compute and return the design matrix.

    Args:
        time: The time array in TJD.
        ydeg: The maximum spherical harmonic degree.
        nt: The number of map temporal components.
        phase0: The phase of the map at `t = 0` in degrees.
    """
    # Instantiate a `starry` map
    map = starry.Map(ydeg=ydeg, udeg=0, reflected=True, nt=nt)

    # Load the SPICE data
    ephemFiles = glob.glob('../data/TESS_EPH_PRE_LONG_2018*.bsp')
    tlsFile = '../data/tess2018338154046-41240_naif0012.tls'
    solarSysFile = '../data/tess2018338154429-41241_de430.bsp'
    #print(spice.tkvrsn('TOOLKIT'))
    for ephFil in ephemFiles:
        spice.furnsh(ephFil)
    spice.furnsh(tlsFile)
    spice.furnsh(solarSysFile)

    # JD time range
    allTJD = time + TJD0
    nT = len(allTJD)
    allET = np.zeros((nT,), dtype=np.float)
    for i, t in enumerate(allTJD):
        allET[i] = spice.unitim(t, 'JDTDB', 'ET')

    # Calculate positions of TESS, the Earth, and the Sun
    # Note that our reference frame is that of `starry`,
    # where `y` points *north*.
    # This is just the rotation {x, y, z} --> {z, x, y}
    # relative to the J200 mean equatorial coordinates.
    tess = np.zeros((3, len(allET)))
    sun = np.zeros((3, len(allET)))
    for i, et in enumerate(allET):
        outTuple = spice.spkezr('Mgs Simulation', et, 'J2000', 'NONE', 'Earth')
        tess[0, i] = outTuple[0][1] * REARTH
        tess[1, i] = outTuple[0][2] * REARTH
        tess[2, i] = outTuple[0][0] * REARTH
        outTuple = spice.spkezr('Sun', et, 'J2000', 'NONE', 'Earth')
        sun[0, i] = outTuple[0][1] * REARTH
        sun[1, i] = outTuple[0][2] * REARTH
        sun[2, i] = outTuple[0][0] * REARTH

    # Compute the linear starry model
    t = (time - time[0]) / (time[-1] - time[0])
    t = 2 * (t - 0.5)
    X = np.empty((nT, map.Ny * map.nt))
    tess_hat = tess / np.sqrt(np.sum(tess ** 2, axis=0))
    sun_hat = sun / np.sqrt(np.sum(sun ** 2, axis=0))
    for i in tqdm(range(len(time))):
        # Rotate the sun into place
        costheta = np.dot(tess_hat[:, i], [0, 0, 1])
        axis = np.cross(tess_hat[:, i], [0, 0, 1])
        sintheta = np.sqrt(np.sum(axis ** 2))
        axis /= sintheta
        theta = 180. / np.pi * np.arctan2(sintheta, costheta)
        R = starry.RAxisAngle(axis, theta)
        source = np.dot(R, sun_hat[:, i])

        # Find the effective rotation matrix that first
        # rotates the earth into phase and then rotates
        # the earth into view for tess
        tx, ty, tz = tess_hat[:, i]
        phase = (2 * np.pi * time[i]) % (2 * np.pi) + phase0
        ty2 = ty ** 2
        tx2 = tx ** 2
        cosp = np.cos(phase)
        sinp = np.sin(phase)
        costheta = (ty2 * (tz - 1) + (2 * tx2 * tz + ty2 * (tz + 1)) * cosp + 
                    2 * tx * (tx2 + ty2) * sinp) / (2 * (tx2 + ty2))
        theta = 180. / np.pi * np.arccos(costheta)
        axis = np.array([
            ty + ty * cosp + (tx * ty * (1 - tz) * sinp) / (tx2 + ty2),
            -2 * tx * cosp + tz * sinp + ((ty2 * (1 - tz)) / (tx2 + ty2) + tz) * sinp,
            tx * ty * (1 - tz) / (tx2 + ty2) - (tx * ty * (1 - tz) * cosp) / (tx2 + ty2) + ty * sinp
        ])
        axis /= np.sqrt(np.sum(axis ** 2))
        map.axis = axis
        X[i] = map.linear_flux_model(t=t[i], theta=theta, source=source)

    if fit_linear_term:
        return X
    else:
        # Let's remove the Y_{0,0} from the design matrix and return
        # the static constant term so that it can be subtracted from the data.
        X00 = np.array(X[:, 0])
        X = np.delete(X, [n * map.Ny for n in range(map.nt)], axis=1)
        return X, X00