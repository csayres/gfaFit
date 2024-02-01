from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS

mjd = 60285
gimg = "gimg-gfa1n-0315"

imgFile = "~/Downloads/gcam/%i/%s.fits"%(mjd,gimg)
ff = fits.open(imgFile)
xyf = fits.open("%s.xyls"%gimg)[1].data

# plt.figure()
# plt.imshow(ff[1].data, origin="lower")
# plt.plot(xyf["x"], xyf["y"], "o", mfc="none", mec="red", ms=5)
# plt.show()

wcs = WCS(fits.open("%s.wcs"%gimg)[0].header)

ff[1].header.update(wcs)


import pdb; pdb.set_trace()