import spiceypy as spice
import glob
from tqdm import tqdm
import numpy as np
import os
import healpy as hp
import starry
assert starry.__version__ == "1.0.0.dev0", \
    "This code requires the `starry` version 1.0.0.dev0."

# Constants
REARTH = 1.0 / 6371.0
TJD0 = 2457000
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"

__all__ = ["design_matrix", "tess_vector", "visibility", "smooth"]


def alm_to_ylm(alm, lmax):
    # Convert the alms to real coefficients
    ylm = np.zeros(lmax ** 2 + 2 * lmax + 1, dtype='float')
    i = 0
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            j = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m))
            if m < 0:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].imag
            elif m == 0:
                ylm[i] = alm[j].real
            else:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].real
            i += 1
    return ylm


def ylm_to_alm(ylm, lmax):
    # Convert the ylms to complex coefficients
    alm = np.zeros(lmax ** 2 + 2 * lmax + 1, dtype='complex128') * np.nan
    i = 0
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            j = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m))
            if np.isnan(alm[j]):
                alm[j] = 0.0
            if m < 0:
                alm[j] += 1j * (ylm[i] / (np.sqrt(2) * (-1) ** m))
            elif m == 0:
                alm[j] += ylm[i]
            else:
                alm[j] += ylm[i] / (np.sqrt(2) * (-1) ** m)
            i += 1
    alm = alm[:np.argmax(np.isnan(alm))]
    return alm


def smooth(map, sigma):
    """Smooth a map in place with a Gaussian kernel."""
    lmax = map.ydeg
    if map._temporal:
        alm = ylm_to_alm(map[:, :, 0], lmax)
    else:
        alm = ylm_to_alm(map[:, :], lmax)
    hp.sphtfunc.smoothalm(alm, sigma=sigma, inplace=True, verbose=False)
    ylm = alm_to_ylm(alm, lmax)
    ylm /= ylm[0]
    if map._temporal:
        map[:, :, 0] = ylm
    else:
        map[:, :] = ylm
    return


def tess_earth_vector(time):
    """Return the cartesian position of TESS relative to the Earth."""
    # Load the SPICE data
    ephemFiles = glob.glob(path + 'TESS_EPH_PRE_LONG_2018*.bsp')
    tlsFile = path + 'tess2018338154046-41240_naif0012.tls'
    solarSysFile = path + 'tess2018338154429-41241_de430.bsp'
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


def design_matrix(time, ydeg=10, nt=2, period=0.9972696, phase0=-56.5, 
                  time0=1325.5, fit_linear_term=False):
    """
    Compute and return the design matrix.

    Args:
        time: The time array in TJD.
        ydeg: The maximum spherical harmonic degree.
        nt: The number of map temporal components.
        phase0: The phase of the map at `t = time0` in degrees.
        time0: The time at which the phase is defined in days.
    """
    # Instantiate a `starry` map
    map = starry.Map(ydeg=ydeg, udeg=0, reflected=True, nt=nt)

    # Load the SPICE data
    ephemFiles = glob.glob(path + 'TESS_EPH_PRE_LONG_2018*.bsp')
    tlsFile = path + 'tess2018338154046-41240_naif0012.tls'
    solarSysFile = path + 'tess2018338154429-41241_de430.bsp'
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
        phase = (2 * np.pi * (time[i] - time0) / period) % (2 * np.pi) + phase0 * np.pi / 180
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


def visibility(time, phase0=-56.5, time0=1325.5, res=100, period=0.9972696):
    """

    """
    # Load the SPICE data
    ephemFiles = glob.glob(path + 'TESS_EPH_PRE_LONG_2018*.bsp')
    tlsFile = path + 'tess2018338154046-41240_naif0012.tls'
    solarSysFile = path + 'tess2018338154429-41241_de430.bsp'
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

    north_pole = np.empty((len(time), 3))
    vernal_eq = np.empty((len(time), 3))
    tess_hat = tess / np.sqrt(np.sum(tess ** 2, axis=0))
    sun_hat = sun / np.sqrt(np.sum(sun ** 2, axis=0))

    # Northern hemisphere
    lonN = np.linspace(-np.pi, np.pi, res)
    latN = np.linspace(1e-2, np.pi / 2, res // 2)
    lonN, latN = np.meshgrid(lonN, latN)
    lonN = lonN.flatten()
    latN = latN.flatten()
    xN = np.sin(np.pi / 2 - latN) * np.cos(lonN - np.pi / 2)
    yN = np.sin(np.pi / 2 - latN) * np.sin(lonN - np.pi / 2)
    zN = np.sqrt(1 - xN ** 2 - yN ** 2)
    R = starry.RAxisAngle([1, 0, 0], -90)
    xN, yN, zN = np.dot(R, np.array([xN, yN, zN]))

    # Southern hemisphere
    lonS = np.linspace(-np.pi, np.pi, res)
    latS = np.linspace(-np.pi / 2, -1e-2, res // 2)
    lonS, latS = np.meshgrid(lonS, latS)
    lonS = lonS.flatten()
    latS = latS.flatten()
    xS = np.sin(np.pi / 2 - latS) * np.cos(lonS - np.pi / 2)
    yS = np.sin(np.pi / 2 - latS) * np.sin(lonS - np.pi / 2)
    zS = np.sqrt(1 - xS ** 2 - yS ** 2)
    R = starry.RAxisAngle([1, 0, 0], 90)
    xS, yS, zS = np.dot(R, np.array([xS, yS, zS]))
    grid = np.hstack((
            np.append(xS, xN).reshape(-1, 1),
            np.append(yS, yN).reshape(-1, 1),
            np.append(zS, zN).reshape(-1, 1)
    )).T
    viz = np.zeros(grid.shape[1])
    lat = np.append(latS, latN)
    lon = np.append((np.pi - lonS), lonN)
    lon[lon > np.pi] -= 2 * np.pi

    for i in tqdm(range(len(time))):
        # Rotate the earth to the current phase in the original frame
        phase = (360. * (time[i] - time0) / period) % 360. + phase0
        R = starry.RAxisAngle([0, 1, 0], phase)
        grid_i = np.dot(R, grid)

        # Rotate the earth and the sun into tess' frame  
        costheta = np.dot(tess_hat[:, i], [0, 0, 1])
        axis = np.cross(tess_hat[:, i], [0, 0, 1])
        sintheta = np.sqrt(np.sum(axis ** 2))
        axis /= sintheta
        theta = 180. / np.pi * np.arctan2(sintheta, costheta)
        R = starry.RAxisAngle(axis, theta)
        assert np.allclose(np.dot(R, tess_hat[:, i]), [0, 0, 1])
        north_pole[i] = np.dot(R, [0, 1, 0])
        vernal_eq[i] = np.dot(R, [0, 0, 1])
        source = np.dot(R, sun_hat[:, i])
        grid_i = np.dot(R, grid_i)

        # Finally, rotate the image so that north always points up
        # This doesn't actually change the integrated flux!
        theta = 180. / np.pi * np.arctan2(north_pole[i, 0], north_pole[i, 1])
        R = starry.RAxisAngle([0, 0, 1], theta)
        north_pole[i] = np.dot(R, north_pole[i])
        vernal_eq[i] = np.dot(R, vernal_eq[i])
        source = np.dot(R, source)
        grid_i = np.dot(R, grid_i)

        # Update the visibility array
        # Weight it by both the cosine of the observer angle
        # and the cosine of the source angle
        gx, gy, gz = grid_i
        cos_s = np.dot(source, grid_i)
        inds = (gz > 0) & (cos_s > 0)
        viz[inds] += gz[inds] * cos_s[inds]

    return lon * 180 / np.pi, lat * 180 / np.pi, viz