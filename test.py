import mapvbvd
import numpy as np
import matplotlib.pyplot as plt

def ifft1d(x, axis=-1):
    return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(x, axes=axis), axis=axis, norm='ortho'), axes=axis)


def sos(x, axis=-1):
    return np.sqrt(np.sum(np.abs(x)**2, axis=axis))


def readKspace(filename):
    twixObj = mapvbvd.mapVBVD(filename)
    twixObj[1].image.flagRemoveOS = False
    kspace = np.squeeze(twixObj[1].image['']) # fe, ch, pe, slc; 3d kspace
    kspace = np.transpose(kspace, (3, 1, 0, 2)) # slc, ch, fe, pe;
    par, ch, fe, pe = kspace.shape
    par_pad, fe_pad = (256-par)//2, 512-fe
    kspace = np.pad(kspace, ((par_pad, par_pad), (0, 0), (fe_pad, 0), (0, 0)))
    return kspace


filename = '/fs03/ab57/kamlesh/Data/Motion/kspace/MRH021_MCAP03042019_MR01/meas_MID00161_FID21338_2__move_T1_mprage_sag_p2_iso_1mm.dat'
kspace = readKspace(filename) # slc, ch, fe, pe: (256, 32, 512, 256)

# --- ifft in slice direction ---- #
kspace = ifft1d(kspace, axis=0) # 2d kspace

slc = 100
kspace_slc = kspace[slc]
img_ch = ifft1d(ifft1d(kspace_slc, axis=-2), axis=-1) 
img = sos(img_ch, axis=0)
plt.imshow(np.abs(img), cmap='gray')
plt.show()

# --- get mask --- #
mask = np.ones(pe, dtype=np.float32)
mask[np.where(np.sum(kspace, axis=(0, 1, 2))==0)] = 0.0
mask = np.expand_dims(mask, axis=(0, 1, 2, -1))



