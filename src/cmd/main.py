"""
   This file is part of fovpreview

   fovpreview is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   any later version.

   fovpreview is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   For a copy of the GNU General Public License see
   <http://www.gnu.org/licenses/>.


   Copyright 2023 Philipp Weber
"""


hipsName = "CDS/P/DSS2/color"

sensors = {
          "APS-C": [22.2, 14.8],
          "VF": [36., 24.]
        }

FLs = [
        50.,
        135.,
        500.,
        800.,
        1000.,
        1200.,
        2350.
        ]

import sys
import io
import os
import tqdm
from multiprocessing.pool import Pool
from math import pi, atan, cos
from warnings import warn
import logging
from argparse import ArgumentParser
import numpy as np
import astroquery.hips2fits
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
from regions import RectangleSkyRegion
from hips import make_sky_image
from hips import WCSGeometry
from PIL import Image

plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'

sys.tracebacklimit = 0

def fmtExc(msg, e):
    excstr = ""
    if len(e.args) == 1:
        excstr = ": {}".format(e.args[0])
    return "{}{}".format(msg, excstr)

class Config:
    __instance = None

    @staticmethod
    def instance():
        if Config.__instance is None:
            Config.__instance = Config()
        return Config.__instance

    def __parseCommandLine(self):
        parser = ArgumentParser()
        parser.add_argument("--outDir", help="Directory for output files", required=True, type=str)
        parser.add_argument("--object", type=str, required=False, help="Object name (resolved by Simbad)")
        parser.add_argument("--ra", type=str, required=False, help="Right ascension of the object")
        parser.add_argument("--dec", type=str, required=False, help="Declination of the object")
        parser.add_argument("--debug", default=False, action="store_true", required=False, help="Enable debug output")
        parser.add_argument("--resx", required=False, help="Horizontal resolution of the output images", default=3840, type=int)
        parser.add_argument("--resy", required=False, help="Vertical resolution of the output images", default=2160, type=int)
        parser.add_argument("--rotation", required=False, default=0., type=float, help="Rotation angle of the sensor in degrees")
        parser.add_argument("--overhead", required=False, default=0.2, type=float, help="Image width overhead")
        parser.add_argument("--dpi", required=False, default=300, type=int, help="Ouput image DPI")
        parser.add_argument("--box-color", required=False, default="red", help="Color of the FOV box")
        parser.add_argument("--clobber", required=False, action="store_true", default=False, help="Overwrite existing output files")
        parser.add_argument("--procs", required=False, default=1, type=int, help="Number of parallel processes")

        self.args = parser.parse_args()

        self.outDir = self.args.outDir
        self.object = self.args.object
        self.ra = self.args.ra
        self.dec = self.args.dec
        self.debug = self.args.debug
        self.resx = self.args.resx
        self.resy = self.args.resy
        self.rotation = self.args.rotation
        self.overhead = self.args.overhead
        self.dpi = self.args.dpi
        self.boxColor = self.args.box_color
        self.clobber = self.args.clobber
        self.procs = self.args.procs

    def __verifyParameters(self):
        if self.rotation < -180 or self.rotation > 180:
            raise ValueError("Rotation must be between -180 and 180")
        if self.resx <= 0:
            raise ValueError("resx must be larger than 0")
        if self.resy <= 0:
            raise ValueError("resy must be larger than 0")
        if self.object is None and self.ra is None and self.dec is None:
            raise ValueError("Need either --object or --ra with --dec")
        if self.ra is not None and self.dec is None:
            raise ValueError("Need --dec with --ra")
        if self.dec is not None and self.ra is None:
            raise ValueError("Need --ra with --dec")

    def __setupDebug(self):
        del sys.tracebacklimit

    def __setupLogger(self):
        self.logger = logging.Logger("Main")

    def __initObjectFromName(self):
        try:
            self.coord = SkyCoord.from_name(self.object).fk5
        except Exception as e:
            raise RuntimeError(fmtExc("Could not find object {}".format(self.object), e))

    def __initObjectFromEquatorial(self):
        self.object = None
        try:
            self.ra = float(self.ra)
            self.dec = float(self.dec)
            self.coord = SkyCoord(self.ra, self.dec, frame="fk5", unit="deg")
            return
        except:
            pass

        try:
            self.coord = SkyCoord(self.ra, self.dec, frame="fk5")
        except Exception as e:
            raise ValueError(fmtExc("Could not understand input coordinates", e))

    def __init__(self):
        self.__parseCommandLine()
        self.__setupLogger()
        self.__verifyParameters()
        if self.debug:
            self.__setupDebug()
        if self.object is not None:
            self.__initObjectFromName()
        else:
            self.__initObjectFromEquatorial()

class Plot():
    def ArcSize(self, s, FL):
        return 206.2648 * s / FL

    def __setupWCS(self):
        diag = np.sqrt(self.sx**2 + self.sy**2)
        rot = Config.instance().rotation / 180. * pi
        beta = atan(self.sx / self.sy)
        alpha = beta - rot
        phi = pi / 2. - beta - rot
        arcx = Angle(self.ArcSize(cos(phi) * diag * 1000, self.FL), unit="arcsec")
        arcy = Angle(self.ArcSize(cos(alpha) * diag * 1000, self.FL), unit="arcsec")


        ARSensor = arcx / arcy
        ARScreen = Config.instance().resx / Config.instance().resy
        if ARSensor < ARScreen:
            # Vertical edge has same length
            self.arcy = arcy
            self.arcx = ARScreen * self.arcy
        else:
            # Horizontal edge has same length
            self.arcx = arcx
            self.arcy = 1./ARScreen * self.arcx

        self.arcx *= ( 1 + Config.instance().overhead )
        self.arcy *= ( 1 + Config.instance().overhead )
        self.scale = self.arcx.degree / Config.instance().resx

        self.__wcs = WCS(header={
            'NAXIS1':           Config.instance().resx,
            'NAXIS2':           Config.instance().resy,
            'WCSAXES':          2,
            'CRPIX1':           Config.instance().resx/2.,
            'CRPIX2':           Config.instance().resy/2.,
            'CDELT1':           -self.scale,
            'CDELT2':           self.scale,
            'CUNIT1':           'deg',
            'CUNIT2':           'deg',
            'CTYPE1':           'RA---TAN',
            'CTYPE2':           'DEC--TAN',
            'CRVAL1':           Config.instance().coord.fk5.ra.degree,
            'CRVAL2':           Config.instance().coord.fk5.dec.degree
            })
        self.__geometry = WCSGeometry(self.__wcs, width=Config.instance().resx, height=Config.instance().resy)

    def __loadImages(self):
        #self.hipsData = make_sky_image(self.__geometry, hipsName, 'jpg', precise=True)
        #self.img = self.hipsData.image
        self.hipsData = hips2fits.query_with_wcs(hips=hipsName, wcs=self.__wcs, get_query_payload=False, format='jpg')
        self.img = self.hipsData
        #self.redImg = np.flipud(self.hipsData[0].data[0,:,:])
        #self.greenImg = np.flipud(self.hipsData[0].data[1,:,:])
        #self.blueImg = np.flipud(self.hipsData[0].data[2,:,:])
        #self.img = np.dstack([self.redImg, self.greenImg, self.blueImg])

    @property
    def angleX(self):
        return Angle(self.ArcSize(self.sx * 1000, self.FL), unit="arcsec")

    @property
    def angleY(self):
        return Angle(self.ArcSize(self.sy * 1000, self.FL), unit="arcsec")

    @property
    def region(self):
        return RectangleSkyRegion(Config.instance().coord, self.angleX, self.angleY, Angle(Config.instance().rotation, unit="deg"))

    def __createPlot(self):
        artist = self.region.to_pixel(self.__wcs).as_artist()
        artist.set_color(Config.instance().boxColor)
        dpi = Config.instance().dpi
        self.fig = plt.figure(dpi=dpi, figsize=(1.299 * Config.instance().resx/dpi, 1.299 * Config.instance().resy/dpi))
        #self.fig = plt.figure(frameon=False)
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot()
        self.ax.axis('off')
        self.ax.imshow(self.img)
        self.ax.add_artist(artist)
        buff = io.BytesIO()
        self.fig.savefig(buff, format='png', bbox_inches = 'tight', pad_inches = 0)
        self.pltImage = Image.open(buff).convert('RGB').resize( (Config.instance().resx, Config.instance().resy), Image.Resampling.LANCZOS)

    def save(self, outFile):
        self.pltImage.save(outFile)


    def __init__(self, FL, sx, sy):
        self.FL = FL
        self.sx = sx
        self.sy = sy
        self.__setupWCS()
        self.__loadImages()
        self.__createPlot()

def doOne(l):
    p = Plot(l[0], l[1], l[2]).save(l[3])

pbar = tqdm.tqdm(total = len(FLs) * len(sensors.keys()))
pool = Pool(Config.instance().procs)
args = []
res = []

def pbarUpdate(bla):
    pbar.update(1)

if not os.path.exists(Config.instance().outDir):
    os.makedirs(Config.instance().outDir)

for fl in FLs:
    for sensor in sensors:
        outFile = os.path.join(Config.instance().outDir, "{:d}mm_{}.jpg".format(int(fl), sensor))
        if os.path.exists(outFile):
            if Config.instance().clobber:
                os.remove(outFile)
            else:
                raise RuntimeError("{} already exists (forgot to set --clobber?".format(outFile))
        args.append([fl, sensors[sensor][0], sensors[sensor][1], outFile])
        res.append(pool.apply_async(doOne, args=[args[-1]], callback=pbarUpdate))

for r in res:
    r.get()

pbar.close()
pbar.join()
