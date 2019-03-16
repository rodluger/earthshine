import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from IPython.display import HTML
from tqdm import tqdm
import glob
import starry
assert starry.__version__ == "1.0.0.dev0", \
    "This code requires the `starry` version 1.0.0.dev0."

# Constants
REARTH = 1.0 / 6371.0
TJD0 = 2457000
cmap = plt.get_cmap("plasma")

__all__ = ["animate"]


# Visualize the orbit
def animate(map, time, phase0=0.0, res=75, interval=75):
    """

    """
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

    # Figure setup
    fig = plt.figure(figsize=(8, 8))
    ax = np.zeros((2, 2), dtype=object)
    ax[0, 0] = plt.subplot(221)
    ax[0, 1] = plt.subplot(222)
    ax[1, 0] = plt.subplot(223, sharex=ax[0, 0], sharey=ax[0, 0])
    ax[1, 1] = plt.subplot(224, sharex=ax[0, 0], sharey=ax[0, 0])
    for axis in [ax[0, 0], ax[1, 0], ax[1, 1]]:
        axis.set_aspect(1)
        axis.set_xlim(-65, 65)
        axis.set_ylim(-65, 65)
        for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
    i = 0

    # Orbit xz
    ax[0, 0].plot(tess[0], tess[2], "k.", ms=1, alpha=0.025)
    txz, = ax[0, 0].plot(tess[0, i], tess[2, i], 'o', color="C0", ms=4)
    norm = 1. / np.sqrt(sun[0, i] ** 2 + sun[2, i] ** 2)
    x = sun[0, i] * norm
    y = sun[2, i] * norm
    theta = 180. / np.pi * np.arctan2(y, x)
    dayxz = Wedge((0, 0), 5, theta - 90, theta + 90, color=cmap(0.8))
    nightxz = Wedge((0, 0), 5, theta + 90, theta + 270, color=cmap(0.0))
    ax[0, 0].add_artist(dayxz)
    ax[0, 0].add_artist(nightxz)
    ax[0, 0].set_ylabel("z", fontsize=16)

    # Orbit xy
    ax[1, 0].plot(tess[0], tess[1], "k.", ms=1, alpha=0.025)
    txy, = ax[1, 0].plot(tess[0, i], tess[1, i], 'o', color="C0", ms=4)
    norm = 1. / np.sqrt(sun[0, i] ** 2 + sun[1, i] ** 2)
    x = sun[0, i] * norm
    y = sun[1, i] * norm
    theta = 180. / np.pi * np.arctan2(y, x)
    dayxy = Wedge((0, 0), 5, theta - 90, theta + 90, color=cmap(0.8))
    nightxy = Wedge((0, 0), 5, theta + 90, theta + 270, color=cmap(0.0))
    ax[1, 0].add_artist(dayxy)
    ax[1, 0].add_artist(nightxy)
    ax[1, 0].set_xlabel("x", fontsize=16)
    ax[1, 0].set_ylabel("y", fontsize=16)
    
    # Orbit zy
    ax[1, 1].plot(tess[2], tess[1], "k.", ms=1, alpha=0.025)
    tzy, = ax[1, 1].plot(tess[2, i], tess[1, i], 'o', color="C0", ms=4)
    norm = 1. / np.sqrt(sun[2, i] ** 2 + sun[1, i] ** 2)
    x = sun[2, i] * norm
    y = sun[1, i] * norm
    theta = 180. / np.pi * np.arctan2(y, x)
    dayzy = Wedge((0, 0), 5, theta - 90, theta + 90, color=cmap(0.8))
    nightzy = Wedge((0, 0), 5, theta + 90, theta + 270, color=cmap(0.0))
    ax[1, 1].add_artist(dayzy)
    ax[1, 1].add_artist(nightzy)
    ax[1, 1].set_xlabel("z", fontsize=16)

    # Render the image
    t = (time - time[0]) / (time[-1] - time[0])
    t = 2 * (t - 0.5)
    Z = np.empty((len(time), res, res))
    north_pole = np.empty((len(time), 3))
    vernal_eq = np.empty((len(time), 3))
    tess_hat = tess / np.sqrt(np.sum(tess ** 2, axis=0))
    sun_hat = sun / np.sqrt(np.sum(sun ** 2, axis=0))
    y = np.array(map[:, :, :])
    for i in tqdm(range(len(time))):
        # Rotate the earth to the current phase in the original frame
        map[:, :, :] = y
        map.axis = [0, 1, 0]
        phase = (360. * time[i]) % 360. + phase0
        map.rotate(phase)

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
        map.axis = axis
        map.rotate(theta)

        # Finally, rotate the image so that north always points up
        # This doesn't actually change the integrated flux!
        map.axis = [0, 0, 1]
        theta = 180. / np.pi * np.arctan2(north_pole[i, 0], north_pole[i, 1])
        map.rotate(theta)
        R = starry.RAxisAngle([0, 0, 1], theta)
        north_pole[i] = np.dot(R, north_pole[i])
        vernal_eq[i] = np.dot(R, vernal_eq[i])
        source = np.dot(R, source)

        # Render the image
        Z[i] = map.render(t=t[i], source=source, res=res)[0]
    
    # Reset the map
    map[:, :, :] = y
    map.axis = [0, 1, 0]

    # Image
    vmin = 0.0
    vmax = np.nanmax(Z)
    cmap.set_under(cmap(vmin))
    image = ax[0, 1].imshow(Z[0], extent=(-1, 1, -1, 1), 
                            origin="lower", cmap=cmap,
                            vmin=vmin, vmax=vmax)
    npl, = ax[0, 1].plot(north_pole[0, 0], north_pole[0, 1], marker=r"$N$", color="r")
    spl, = ax[0, 1].plot(-north_pole[0, 0], -north_pole[0, 1], marker=r"$S$", color="b")
    veq, = ax[0, 1].plot(vernal_eq[0, 0], vernal_eq[0, 1], marker=r"$V$", color="y")
    if north_pole[0, 2] > 0:
        npl.set_visible(True)
        spl.set_visible(False)
    else:
        npl.set_visible(False)
        spl.set_visible(True)
    if vernal_eq[0, 2] > 0:
        veq.set_visible(True)
    else:
        veq.set_visible(False)
    ax[0, 1].axis("off")
    ax[0, 1].set_xlim(-1.1, 1.1)
    ax[0, 1].set_ylim(-1.1, 1.1)

    # Function to animate each frame
    def update(i):
        # Update orbit
        txz.set_xdata(tess[0, i])
        txz.set_ydata(tess[2, i])
        norm = 1. / np.sqrt(sun[0, i] ** 2 + sun[2, i] ** 2)
        x = sun[0, i] * norm
        y = sun[2, i] * norm
        theta = 180. / np.pi * np.arctan2(y, x)
        dayxz.set_theta1(theta - 90)
        dayxz.set_theta2(theta + 90)
        nightxz.set_theta1(theta + 90)
        nightxz.set_theta2(theta + 270)
        txy.set_xdata(tess[0, i])
        txy.set_ydata(tess[1, i])
        norm = 1. / np.sqrt(sun[0, i] ** 2 + sun[1, i] ** 2)
        x = sun[0, i] * norm
        y = sun[1, i] * norm
        theta = 180. / np.pi * np.arctan2(y, x)
        dayxy.set_theta1(theta - 90)
        dayxy.set_theta2(theta + 90)
        nightxy.set_theta1(theta + 90)
        nightxy.set_theta2(theta + 270)
        tzy.set_xdata(tess[2, i])
        tzy.set_ydata(tess[1, i])
        norm = 1. / np.sqrt(sun[2, i] ** 2 + sun[1, i] ** 2)
        x = sun[2, i] * norm
        y = sun[1, i] * norm
        theta = 180. / np.pi * np.arctan2(y, x)
        dayzy.set_theta1(theta - 90)
        dayzy.set_theta2(theta + 90)
        nightzy.set_theta1(theta + 90)
        nightzy.set_theta2(theta + 270)
        image.set_data(Z[i])
        npl.set_xdata(north_pole[i, 0])
        npl.set_ydata(north_pole[i, 1])
        spl.set_xdata(-north_pole[i, 0])
        spl.set_ydata(-north_pole[i, 1])
        veq.set_xdata(vernal_eq[i, 0])
        veq.set_ydata(vernal_eq[i, 1])
        if north_pole[i, 2] > 0:
            npl.set_visible(True)
            spl.set_visible(False)
        else:
            npl.set_visible(False)
            spl.set_visible(True)
        if vernal_eq[i, 2] > 0:
            veq.set_visible(True)
        else:
            veq.set_visible(False)
        return txz, dayxz, nightxz, txy, dayxy, nightxy, \
               tzy, dayzy, nightzy, image, npl, spl, veq

    # Generate the animation
    ani = FuncAnimation(fig, update, frames=len(time), interval=interval, 
                        blit=False)
    
    try:
        if 'zmqshell' in str(type(get_ipython())):
            plt.close()
            display(HTML(ani.to_html5_video()))
        else:
            raise NameError("")
    except NameError:
        plt.show()
        plt.close()

    return np.nansum(Z, axis=(1, 2))


if __name__ == "__main__":
    import starry_beta
    import starry
    map = starry_beta.Map(15)
    map.load_image("earth")
    yearth = np.array(map[:, :])

    map = starry.Map(15, reflected=True, nt=1)
    map[:, :, 0] = yearth
    time = np.linspace(1325, 1335, 250)
    animate(map, time, res=50, interval=200)