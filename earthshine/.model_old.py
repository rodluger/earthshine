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

__all__ = ["design_matrix"]


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
    tess = np.zeros((3, len(allET)))
    sun = np.zeros((3, len(allET)))
    for i, et in enumerate(allET):
        outTuple = spice.spkezr('Mgs Simulation', et, 'J2000', 'NONE', 'Earth')
        tess[0, i] = outTuple[0][0] * REARTH
        tess[1, i] = outTuple[0][1] * REARTH
        tess[2, i] = outTuple[0][2] * REARTH
        outTuple = spice.spkezr('Sun', et, 'J2000', 'NONE', 'Earth')
        sun[0, i] = outTuple[0][0] * REARTH
        sun[1, i] = outTuple[0][1] * REARTH
        sun[2, i] = outTuple[0][2] * REARTH

    # Compute the linear starry model
    t = (time - time[0]) / (time[-1] - time[0])
    t = 2 * (t - 0.5)
    X = np.empty((nT, map.Ny * map.nt))
    for i in tqdm(range(len(time))):
        # Find the rotation matrix `R` that rotates TESS onto the +z axis
        r = np.sqrt(np.sum(tess[:, i] ** 2))
        costheta = np.dot(tess[:, i], [0, 0, r])
        axis = np.cross(tess[:, i], [0, 0, r])
        sintheta = np.sqrt(np.sum(axis ** 2))
        axis /= sintheta
        R = starry.RAxisAngle(axis, 180. / np.pi * np.arctan2(sintheta, costheta))
        
        # Rotate into this new frame. The Earth is still
        # at the origin, TESS is along the +z axis, and
        # the Sun is at `source`.
        nx, ny, nz = np.dot(R, [0, 0, 1])
        source = np.dot(R, sun[:, i])
        source /= np.sqrt(np.sum(source ** 2, axis=0))

        # We need to rotate the map of the Earth so the
        # north pole is at (nx, ny, nz). We also need to
        # rotate the Earth about the pole to get it to the
        # correct phase. We'll do this with a compound
        # rotation by an angle `theta` about the axis computed
        # below. 
        phase = 2 * np.pi / period * (time[i]) + np.pi / 180. * phase0
        cosphase = np.cos(phase)
        sinphase = np.sin(phase)
        map.axis = [nz + nz * cosphase + nx * sinphase,
                    (1 + ny) * sinphase,
                    -nx - nx * cosphase + nz * sinphase]
        costheta = 0.5 * (-1 + ny + cosphase + ny * cosphase)
        sintheta = np.sqrt(1 - costheta ** 2)
        theta = 180 / np.pi * np.arctan2(sintheta, costheta)
        X[i] = map.linear_flux_model(t=t[i], theta=theta, source=source)

    if fit_linear_term:
        return X
    else:
        # Let's remove the Y_{0,0} from the design matrix and return
        # the static constant term so that it can be subtracted from the data.
        X00 = np.array(X[:, 0])
        X = np.delete(X, [n * map.Ny for n in range(map.nt)], axis=1)
        return X, X00