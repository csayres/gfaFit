import pandas
from astropy.io import fits
import os
import numpy
from subprocess import Popen, TimeoutExpired, PIPE
from astropy.table import Table
from astroscrappy import detect_cosmics
import sep
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
from astropy.wcs import WCS
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from shutil import copyfile
import glob
import seaborn as sns
from coordio.transforms import arg_nearest_neighbor
from coordio.conv import guideToTangent, tangentToWok
from astropy.time import Time

from coordio import defaults
from coordio.sky import ICRS, Observed
from coordio.telescope import Field, FocalPlane
from coordio.wok import Wok
from coordio.site import Site

gfaCoords = pandas.read_csv("/Users/csayres/code/fps_calibrations/apo/wok_calibs/sloanFlatCMM/gfaCoords.csv")
GAIA_EPOCH = 2457206

gfaRot = {
    1 : 60,
    2 : 0,
    3 : -60,
    4 : -60*2,
    5 : -60*3,
    6 : -60*4
}

## create test pixels
_xpix = []
_ypix = []
for xpix in range(0,2048,200):
    for ypix in range(0,2048,200):
        _xpix.append(xpix)
        _ypix.append(ypix)
_xpix = numpy.array(_xpix)
_ypix = numpy.array(_ypix)

SIP_COEFFS = [
    "A_0_2" ,
    "A_1_1" ,
    "A_2_0" ,
    "B_0_2" ,
    "B_1_1" ,
    "B_2_0" ,
    "AP_0_0",
    "AP_0_1",
    "AP_0_2",
    "AP_1_0",
    "AP_1_1",
    "AP_2_0",
    "BP_0_0",
    "BP_0_1",
    "BP_0_2",
    "BP_1_0",
    "BP_1_1",
    "BP_2_0"
]


def radec2wokxy(ra, dec, coordEpoch, waveName, raCen, decCen, obsAngle,
                obsSite, obsTime, focalScale=None, pmra=None, pmdec=None, parallax=None,
                radVel=None, pressure=None, relativeHumidity=0.5,
                temperature=10):
    """
    Convert from ra, dec ICRS coords to a flat-wok XY in mm.  At obsAngle=0
    wok +y is a aligned with +dec, and wok +x is aligned with +ra

    Question for José, do we need to enforce a time scale?  I think everything
    defaults to UTC.

    Parameters
    ------------
    ra : numpy.ndarray
        Right ascension in ICRS, degrees
    dec : numpy.ndarray
        Declination in ICRS, degrees
    coordEpoch : float
        A TDB Julian date, the epoch for the input
        coordinates (from the catalog). Defaults to J2000.
    waveName : str, or numpy.ndarray
        Array elements must be "Apogee", "Boss", or "GFA" strings
    raCen : float
        Right ascension of field center, in degrees
    decCen : float
        Declination of field center, in degrees.
    obsAngle : float
        Position angle for observation.  Angle is measured from North
        through East to wok +y. So obsAngle of 45 deg, wok +y points NE.
    obsSite : str
        Either "APO" or "LCO"
    obsTime : float
        TDB Julian date.  The time at which these coordinates will be observed
        with the FPS.
    focalScale : float or None
        Scale factor for conversion between focal and wok coords, found
        via dither analysis.  Defaults to value in defaults.SITE_TO_SCALE
    pmra : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr. Must be a true angle, i.e, it must include the
        ``cos(dec)`` term.  Defaults to 0.
    pmdec : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr.  Defaults to 0.
    parallax : numpy.ndarray
        A 1D array with the parallax for the N targets, in milliarcsec.
        Defaults to 0.
    radVel : numpy.ndarray
        A 1D array with the radial velocity in km/s, positive when receding.
        Defaults to 0.
    pressure : float
        The atmospheric pressure at the site, in millibar (same as hPa). If
        not provided the pressure will be calculated using the altitude
        :math:`h` and the approximate expression

        .. math::

            p \sim -p_0 \exp\left( \dfrac{g h M} {T_0 R_0} \right)

        where :math:`p_0` is the pressure at sea level, :math:`M` is the molar
        mass of the air, :math:`R_0` is the universal gas constant, and
        :math:`T_0=288.16\,{\rm K}` is the standard sea-level temperature.
    relativeHumidity : float
        The relative humidity, in the range :math:`0-1`. Defaults to 0.5.
    temperature : float
        The site temperature, in degrees Celsius. Defaults to
        :math:`10^\circ{\rm C}`.

    Returns
    ---------
    xWok : numpy.ndarray
        x wok coordinate, mm
    yWok : numpy.ndarray
        y wok coordinate, mm
    fieldWarn : numpy.ndarray
        boolean array.  Where True the coordinate converted should be eyed with
        suspicion.  (It came from very far off axis).
    hourAngle : float
        hour angle of field center in degrees
    positionAngle : float
        position angle of field center in degrees
    """
    nCoords = len(ra)

    # first grab the correct wavelengths for fibers
    wavelength = numpy.zeros(nCoords)

    if isinstance(waveName, str):
        # same wl for all coords
        wavelength += defaults.INST_TO_WAVE[waveName]
    else:
        assert len(waveName) == nCoords
        for ii, ft in enumerate(waveName):
            wavelength[ii] = defaults.INST_TO_WAVE[ft]

    site = Site(
        obsSite, pressure=pressure,
        temperature=temperature, rh=relativeHumidity
    )
    site.set_time(obsTime)

    # first determine the field center in observed coordinates
    # use the guide wavelength for field center
    # epoch not needed, no propermotions, etc (josé verify?)
    icrsCen = ICRS([[raCen, decCen]])
    obsCen = Observed(icrsCen, site=site, wavelength=defaults.INST_TO_WAVE["GFA"])

    radec = numpy.array([ra, dec]).T

    icrs = ICRS(
        radec, epoch=coordEpoch, pmra=pmra, pmdec=pmdec,
        parallax=parallax, rvel=radVel
    )

    # propogate propermotions, etc
    icrs = icrs.to_epoch(obsTime, site=site)
    if focalScale is None:
        focalScale = defaults.SITE_TO_SCALE[obsSite]

    obs = Observed(icrs, site=site, wavelength=wavelength)
    field = Field(obs, field_center=obsCen)
    focal = FocalPlane(field,
                       wavelength=wavelength,
                       site=site,
                       fpScale=focalScale,
                       use_closest_wavelength=True)
    wok = Wok(focal, site=site, obsAngle=obsAngle)

    output = (
        wok[:, 0], wok[:, 1], focal.field_warn,
        float(obsCen.ha), float(obsCen.pa)
    )
    return output


def sipDistort(xs, ys, rev=False, order=2):
    # https://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf
    # xs, ys are pixels
    df = pandas.read_csv("wcs_median.csv")
    keys = list(df["index"])
    vals = list(df["0"])
    df = {}
    for key,val in zip(keys,vals):
        df[key] = val

    if hasattr(xs, "__len__"):
        dx = numpy.zeros(len(xs))
        dy = numpy.zeros(len(ys))
        xs = numpy.array(xs)
        ys = numpy.array(ys)
    else:
        dx = 0
        dy = 0

    xs = xs - df["CRPIX1"]
    ys = ys - df["CRPIX2"]

    for p in range(order+1):
        for q in range(order+1):
            x_coeff = "A_%i_%i"%(p,q)
            y_coeff = "B_%i_%i"%(p,q)

            if x_coeff in df:
                dx += df[x_coeff] * xs**p * ys**q

            if y_coeff in df:
                dy += df[y_coeff] * xs**p * ys**q

    return dx, dy


def extract(imgData):
    imgData = numpy.array(imgData, dtype=float)
    crmask, imgData = detect_cosmics(imgData)
    bkg = sep.Background(imgData)
    imgData = imgData - bkg.back()
    objs = sep.extract(imgData, 5.5, err=bkg.globalrms)
    objs = pandas.DataFrame(objs)
    objs["fwhm"] = 2 * numpy.sqrt(numpy.log(2)*(objs.a**2+objs.b**2))
    return objs


def queryGaia(ra, dec, radius=0.16, magLimit=18):
    sql = "psql -c 'SELECT * FROM catalogdb.gaia_dr2_source_g19 WHERE q3c_radial_query(ra, dec, %.4f, %.4f, %.4f) AND phot_g_mean_mag < %.2f' -d 'postgresql://sdss_user@sdss5-db/chernodb' -a"%(ra,dec,radius,magLimit)
    proc = Popen(["ssh", "sdss5-hub", sql], stdout=PIPE, stderr=PIPE)

    try:
        outs, errs = proc.communicate(timeout=15)
    except TimeoutExpired:
        print("timed out")
        proc.kill()
        outs, errs = proc.communicate()
        return None

    outs = outs.decode("utf-8")
    columns = ["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag"]

    outlines = outs.split("\n")
    outlines = outlines[3:-4]
    outArr = numpy.zeros((len(outlines),len(columns)))

    for ii, line in enumerate(outlines):
        print(ii, line)
        vals = []
        for cell in line.split("|"):
            try:
                vals.append(float(cell.strip()))
            except:
                vals.append(numpy.nan)
        outArr[ii,:] = vals

    df = pandas.DataFrame(outArr, columns=columns)
    return df


def getGimgFile(mjd, gfaNum, imgNum):
    imgStr = str(imgNum).zfill(4)
    gimg = "gimg-gfa%in-%s"%(gfaNum, imgStr)
    imgFile = "/Volumes/futa/apo/data/gcam/%i/%s.fits"%(mjd,gimg)
    return imgFile


def solveCamera(mjd, gfaNum, imgNum, ax=None): #gfaNum, mjd, imgNum):
    # baseDir = "/users/csayres/gfaFit/astronet/%i"%mjd
    # if not os.path.exists(baseDir):
    #     os.mkdir(baseDir)
    # fitspath = "/data/gcam/%i/gimg"
    astNetFile = "xylsTemp"
    imgFile = getGimgFile(mjd, gfaNum, imgNum)

    ff = fits.open(imgFile)
    dateObs = Time(ff[1].header["DATE-OBS"], format="iso", scale="tai")
    objs = extract(ff[1].data)

    objs = objs[objs.peak < 55000]
    objs = objs[objs.peak > 500]
    objs = objs[(objs.x > 50) & (objs.x < 2000)]
    objs = objs[(objs.y > 50) & (objs.y < 2000)]
    objs["gfaNum"] = gfaNum
    objs["imgNum"] = imgNum
    objs["mjd"] = mjd
    objs["fwhm"] = 2 * numpy.sqrt(numpy.log(2)*(objs.a**2+objs.b**2))
    objs["exptime"] = ff[1].header["EXPTIME"]
    objs["jd"] = dateObs.jd
    objs["raBore"] = ff[1].header["RA"]
    objs["decBore"] = ff[1].header["DEC"]
    objs["ipa"] = ff[1].header["IPA"]
    objs["az"] = ff[1].header["AZ"]
    objs["alt"] = ff[1].header["ALT"]
    objs["raField"] = ff[1].header["RAFIELD"]
    objs["decField"] = ff[1].header["DECFIELD"]
    objs["paField"] = ff[1].header["FIELDPA"]

    # dx,dy = sipDistort(objs.x.to_numpy(),objs.y.to_numpy())
    # objs["xOrig"] = objs["x"]
    # objs["yOrig"] = objs["y"]
    # objs["xDist"] = objs.x + dx
    # objs["yDist"] = objs.y + dy

    objs.to_csv("objs_%i_%i_%i.csv"%(mjd,gfaNum,imgNum), index=False)

    print("\n\nn objs %i %i %i \n\n"%(imgNum, gfaNum, len(objs)))


    for centType in ["Tweak", "NoTweak"]:
        if centType == "Tweak":
            netFlag = "--tweak-order 2"
        else:
            netFlag = "--no-tweak"
        crCen = "--crpix-cen"
        newFileName = "wcs_%s_%i_%i_%i.wcs"%(centType,mjd,gfaNum, imgNum)

        xyls = objs[["x", "y", "flux"]].copy()
        xyls = Table.from_pandas(xyls)
        xyls.write("%s.xyls"%astNetFile, format="fits", overwrite=True)
        raBore = ff[1].header["RA"]
        decBore = ff[1].header["DEC"]

        proc = Popen(["scp", "%s.xyls"%astNetFile, "sdss5-hub:/home/csayres/gfaFit/astronet"], stdout=PIPE, stderr=PIPE)

        outs, errs = proc.communicate(timeout=15)
        outs = outs.decode("utf-8").splitlines()
        errs = errs.decode("utf-8").splitlines()
        print("outs", outs)

        shellCmd = "/home/sdss5/software/astrometrynet/0.93/solver/solve-field --backend-config /home/sdss5/software/cherno/main/cherno/etc/astrometrynet_APO.cfg --width 2048 --height 2048 --sort-column flux --ra %.2f --dec %.2f --radius 2 --scale-low 0.1944 --scale-high 0.2376 --scale-units arcsecperpix --overwrite --no-plots %s %s --cpulimit 15 --odds-to-solve 1000000000 --dir /home/csayres/gfaFit/astronet /home/csayres/gfaFit/astronet/%s.xyls"%(raBore, decBore, netFlag, crCen, astNetFile)

        proc = Popen(["ssh", "sdss5-hub", shellCmd], stdout=PIPE, stderr=PIPE)
        outs, errs = proc.communicate(timeout=90)
        outs = outs.decode("utf-8")
        if "Field 1 solved: writing to file" not in outs:
            print("field failed", gfaNum, imgNum, centType)
            return None
        # errs = errs.decode("utf-8").splitlines()
        # print("outs", outs)

        proc = Popen(["scp", "sdss5-hub:/home/csayres/gfaFit/astronet/%s.wcs"%astNetFile, "."], stdout=PIPE, stderr=PIPE)

        outs, errs = proc.communicate(timeout=15)
        outs = outs.decode("utf-8").splitlines()
        errs = errs.decode("utf-8").splitlines()
        print("outs", outs)

        copyfile("%s.wcs"%astNetFile, newFileName)


def processMany():
    # queryGaia(60,60)
    # solveCamera(60285, 6, 461)
    # mjd = 60285
    # imgNums = [
    #     312,
    #     337,
    #     370,
    #     400,
    #     432,
    #     465,
    #     492,
    # ]

    mjd = 60338
    imgNums = [434, 442, 452, 462]

    # old num 461
    for imgNum in imgNums:
        # fig, axs = plt.subplots(2,3,figsize=(10,5))
        # axs = axs.flatten()
        for ii, gfaID in enumerate(range(1,7)):
            solveCamera(mjd, gfaID, imgNum, ax=None)
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig("dist_%i_%i.png"%(mjd, imgNum), dpi=250)
        # plt.close("all")
    # raDeg = 0
    # decDeg = 0

    # cmd = "/home/sdss5/software/astrometrynet/0.93/solver/solve-field --backend-config /home/sdss5/software/cherno/main/cherno/etc/astrometrynet_APO.cfg --width 2048 --height 2048 --sort-column flux --ra %.2f --dec %.2f --radius 2 --scale-low 0.1944 --scale-high 0.2376 --scale-units arcsecperpix --overwrite --cpulimit 15 --odds-to-solve 1000000000 --dir /home/csayres/gfaFit/astronet/%i /data/gcam/%i/astrometry/gimg-gfa%in-%s.xyls"%(raDeg,decDeg,mjd,mjd,gfaNum,imgNumStr)


# gaia_stars = pandas.read_sql("SELECT * FROM catalogdb.gaia_dr2_source_g19 WHERE q3c_radial_query(ra, dec, 60, 60, 0.16) AND phot_g_mean_mag < 18", "postgresql://sdss_user@sdss5-db/chernodb")
# def allWCS():
#     wcsfiles = glob.glob("wcs*.wcs")
#     dd = {}
#     for distCoeff in SIP_COEFFS:
#         dd[distCoeff] = []
#     dd["CRPIX1"] = []
#     dd["CRPIX2"] = []

#     gfaNum = []
#     imgNum = []
#     mjd = []
#     for f in wcsfiles:
#         junk, _runType, _mjd, _gfaNum, _imgNum = f.strip(".wcs").split("_")
#         gfaNum.append(int(_gfaNum))
#         imgNum.append(int(_imgNum))
#         mjd.append(int(_mjd))
#         wcs = WCS(open(f).read())
#         hdr = wcs.to_header(relax=True)
#         for key in dd.keys():
#             if key in hdr:
#                 dd[key].append(hdr[key])
#             else:
#                 dd[key].append(0)

#     dd["gfaNum"] = gfaNum
#     dd["imgNum"] = imgNum
#     dd["mjd"] = mjd
#     dd["runType"] = _runType
#     df = pandas.DataFrame(dd)
#     df.to_csv("allwcs.csv")

#     # wcsMed = df.median().reset_index()
#     # wcsMed.to_csv("wcs_median.csv", index=False)

#     # # import pdb; pdb.set_trace()
#     # # mean = df.median().to_csv("sip_median_all.csv")
#     # # for gfaNum in range(1,7):
#     # #     mean = df[df.gfaNum==gfaNum].median().to_csv("sip_median_%i.csv"%gfaNum)
#     # # import pdb; pdb.set_trace()

#     # centFiles = glob.glob("objs_*.csv")
#     # dfList = []
#     # for centFile in centFiles:
#     #     dfList.append(pandas.read_csv(centFile, index_col=0))

#     # df = pandas.concat(dfList)
#     # xs = df.x.to_numpy()
#     # ys = df.y.to_numpy()
#     # dx,dy = sipDistort(xs,ys)

#     # plt.figure(figsize=(8,8))
#     # ax = plt.gca()
#     # ax.quiver(xs,ys,dx,dy,angles="xy", units="xy", scale=.02)
#     # # ax.set_title("GFA %i  IPA %.1f"%(gfaNum,ff[1].header["IPA"]))
#     # ax.set_aspect("equal")
#     # ax.set_yticks([])
#     # ax.set_xticks([])
#     # plt.show()

#     # import pdb; pdb.set_trace()


def compileWCS():
    wcsFiles = glob.glob("wcs_*.wcs")
    # dfSip = pandas.read_csv("wcs_median.csv")

    ddList = []
    for wcsFile in wcsFiles:
        ff = fits.open(wcsFile)
        tokens = wcsFile.split("_")
        mode = str(tokens[1])
        mjd = int(tokens[2])
        gfaNum = int(tokens[3])
        imgNum = int(tokens[4].strip(".wcs"))
        dd = dict(ff[0].header)
        dd["runType"] = mode
        dd["mjd"] = mjd
        dd["gfaNum"] = gfaNum
        dd["imgNum"] = imgNum
        del dd["HISTORY"]
        del dd["COMMENT"]

        wcs = WCS(ff[0].header)

        racen, deccen = wcs.pixel_to_world_values([[1024.5, 1024.5]])[0]

        dd["racen"] = racen
        dd["deccen"] = deccen

        # dra = (racen - racen1)/numpy.cos(numpy.radians(deccen))
        # ddec = deccen - deccen1
        # cenErr = numpy.sqrt(dra**2+ddec**2)*3600.
        # dd["cenErr"] = cenErr

        ddList.append(dd)

    df = pandas.DataFrame(ddList)
    df.to_csv("wcs_all.csv", index=False)


def getAllGaia():
    dfAll = pandas.read_csv("wcs_all.csv")
    dfAll = dfAll[dfAll.runType=="Tweak"]

    dfList = []
    for mjd, gfaNum, imgNum, racen, deccen in dfAll[["mjd", "gfaNum", "imgNum", "racen", "deccen"]].to_numpy():
        df = queryGaia(racen,deccen)
        df["mjd"] = int(mjd)
        df["gfaNum"] = int(gfaNum)
        df["imgNum"] = int(imgNum)
        df.to_csv("gaia_%i_%i_%i.csv"%(mjd,gfaNum,imgNum), index=False)
        dfList.append(df)

    df = pandas.concat(dfList)
    df.to_csv("gaia_all.csv", index=False)


def matchCentsWCSGaia(mjd):
    gaia = pandas.read_csv("gaia_all.csv")
    gaia = gaia[gaia.phot_g_mean_mag <= 18]
    cents = pandas.concat([pandas.read_csv(x) for x in glob.glob("objs*.csv")])
    imgNums = list(set(cents.imgNum))
    dfList = []
    for runType in ["Tweak", "NoTweak"]:
        for gfaNum in range(1,7):
            for imgNum in imgNums:
                _c = cents[(cents.imgNum==imgNum) & (cents.gfaNum==gfaNum)].reset_index(drop=True)
                _g = gaia[(gaia.imgNum==imgNum) & (gaia.gfaNum==gfaNum)].reset_index(drop=True)
                xyCents = _c[["x", "y"]].to_numpy()
                raDecGaia = _g[["ra", "dec"]].to_numpy()

                wcsFile = "wcs_%s_%i_%i_%i.wcs"%(runType, mjd,gfaNum,imgNum)
                ff = fits.open(wcsFile)
                wcs = WCS(ff[0].header)
                import pdb; pdb.set_trace()

                xyGaia = numpy.array(wcs.world_to_pixel_values(raDecGaia))
                _g["xWCS"] = xyGaia[:,0]
                _g["yWCS"] = xyGaia[:,1]
                raDecCents = numpy.array(wcs.pixel_to_world_values(xyCents))
                _c["raWCS"] = raDecCents[:,0]
                _c["decWCS"] = raDecCents[:,1]


                # ff = fits.open(getGimgFile(mjd,gfaNum,imgNum))
                ipa = _c["ipa"].to_numpy()[0] #ff[1].header["IPA"]
                # print("altitude", ff[1].header["ALT"])

                plt.figure(figsize=(8,8))
                plt.plot(xyCents[:,0],xyCents[:,1],"o", mfc="none", mec="black")

                # plt.scatter(xyCents[:,0], xyCents[:,1], c=flux)
                # plt.colorbar(shrink=0.25)

                plt.plot(xyGaia[:,0], xyGaia[:,1], "xr")

                plt.axis("equal")
                plt.xlim([0,2048])
                plt.ylim([0,2048])
                if runType == "Tweak":
                    sip = "SIP=True"
                else:
                    sip = "SIP=False"


                # find matches
                gimg_scale = 0.216 # arcsec per pixel
                matches, indices, minDists = arg_nearest_neighbor(xyCents, xyGaia)
                xyGaiaMatch = xyGaia[indices]
                raDecGaiaMatch = raDecGaia[indices]
                _gMatch = _g.iloc[indices].reset_index(drop=True)

                df = pandas.concat([_c,_gMatch], axis=1)
                # remove duplicated columns
                df = df.loc[:,~df.columns.duplicated()].copy()
                # import pdb; pdb.set_trace()
                rms = gimg_scale * numpy.sqrt(numpy.mean(minDists**2))

                plt.title('GFA=%i imgNum=%i  IPA=%.0f %s RMS=%.1f"'%(gfaNum, imgNum, ipa, sip, rms))
                plt.savefig("cent_vs_gaia_%i_%i_%i_%s.png"%(mjd, gfaNum, imgNum, runType), dpi=200)
                plt.close("all")

                # df = pandas.DataFrame({
                #     "xCent": xyCents[:,0],
                #     "yCent": xyCents[:,1],
                #     "xGaia": xyGaiaMatch[:,0],
                #     "yGaia": xyGaiaMatch[:,1],
                #     "raCent": raDecCents[:,0],
                #     "decCent": raDecCents[:,1],
                #     "raGaia": raDecGaiaMatch[:,0],
                #     "decGaia": raDecGaiaMatch[:,1]
                # })
                # df["gfaNum"] = gfaNum
                # df["mjd"] = mjd
                # df["imgNum"] = imgNum
                df["runType"] = runType
                # df["ipa"] = ipa
                dfList.append(df)

                # import pdb; pdb.set_trace()
                # plt.figure()
                # plt.hist(minDists*gimg_scale, bins=20)
                # plt.title("GFA: %i  imgNum: %i  IPA: %.0f"%(gfaNum, imgNum, ipa))

                # plt.figure(figsize=(8,8))
                # plt.imshow(ff[1].data, origin="lower")
                # plt.plot(xyCents[:,0], xyCents[:,1], 'o', mfc="none", mec="red")
                # plt.title("GFA: %i  imgNum: %i  IPA: %.0f"%(gfaNum, imgNum, ipa))

    df = pandas.concat(dfList)
    df.to_csv("objs_gaia_match_all.csv", index=False)


def plotWCSErr():
    df = pandas.read_csv("objs_gaia_match_all.csv")
    df["dra"] = (df.raWCS - df.ra)/numpy.cos(numpy.radians(df.dec))
    df["ddec"] = df.decWCS - df.dec
    df = df[df.runType=="NoTweak"]
    skyRMS = numpy.sqrt(numpy.mean(df.dra**2+df.ddec**2))*3600

    plt.figure(figsize=(8,8))
    plt.quiver(df.ra,df.dec,df.dra,df.ddec, angles="xy", units="xy")
    plt.axis("equal")
    plt.xlabel("ra (deg)")
    plt.ylabel("dec (deg)")
    plt.title("sky rms: %.1f arcsec"%skyRMS)

    xt, yt = guideToTangent(df.x.to_numpy(), df.y.to_numpy())
    df["xtCent"] = xt
    df["ytCent"] = yt

    xt, yt = guideToTangent(df.xWCS.to_numpy(), df.yWCS.to_numpy())
    df["xtGaia"] = xt
    df["ytGaia"] = yt

    dfList = []
    for gfaNum in range(1,7):
        g = gfaCoords[gfaCoords.id==gfaNum]

        _df = df[df.gfaNum==gfaNum].reset_index(drop=True)
        zt = numpy.zeros(len(_df))
        b = g[["xWok","yWok","zWok"]].to_numpy()
        iHat = g[["ix", "iy", "iz"]].to_numpy()
        jHat = g[["jx", "jy", "jz"]].to_numpy()
        kHat = g[["kx", "ky", "kz"]].to_numpy()
        for x in ["Cent", "Gaia"]:
            xw,yw,zw = tangentToWok(
                _df["xt%s"%x].to_numpy(),
                _df["yt%s"%x].to_numpy(),
                zt,
                b, iHat, jHat, kHat
            )
            _df["xWok%s"%x] = xw
            _df["yWok%s"%x] = yw
        dfList.append(_df)

    df = pandas.concat(dfList)
    for x in ["Cent", "Gaia"]:
        xw = -1*df["xWok%s"%x].to_numpy()
        yw = df["yWok%s"%x].to_numpy()

        radIPA = numpy.radians(df.ipa.to_numpy())
        cosIPA = numpy.cos(radIPA)
        sinIPA = numpy.sin(radIPA)

        xm = xw*cosIPA - yw*sinIPA
        ym = xw*sinIPA + yw*cosIPA

        df["xMirror%s"%x] = xm
        df["yMirror%s"%x] = ym

    dxm = df.xMirrorCent - df.xMirrorGaia
    dym = df.yMirrorCent - df.yMirrorGaia
    rms = numpy.sqrt(numpy.mean(dxm**2+dym**2))*1000
    plt.figure(figsize=(8,8))
    plt.quiver(df.xMirrorGaia,df.yMirrorGaia,dxm,dym, angles="xy", units="xy")
    plt.axis("equal")
    plt.xlabel("mirror x (mm)")
    plt.ylabel("mirror y (mm)")
    plt.title("focal plane rms: %.1f um"%(rms))

    # import pdb; pdb.set_trace()
    ## convert to mirror coordinates


    plt.show()



if __name__ == "__main__":
    # processMany()
    # compileWCS()
    # getAllGaia()
    matchCentsWCSGaia(60338)
    plotWCSErr()

