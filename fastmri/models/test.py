import torch
from mocoLayer import MocoSimLayer, UndersampleLayer
import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import kornia

def ks2img(x):
    img = np.array(fastmri.ifft2c(x))
    img_cplx = img[..., 0] + 1j*img[..., 1]
    img_rss = np.squeeze(np.sqrt(np.sum(np.abs(img_cplx)**2, 1)))
    return img_rss
      
fname = '/mnt/c/Users/kamle/Documents/dataSets/fastMri/brain/multicoil_val/file_brain_AXFLAIR_200_6002572.h5'
hf = h5py.File(fname, 'r')
kspace = hf["kspace"][2:3]
kspace = np.expand_dims(kspace, -1)
kspace = np.concatenate((np.real(kspace), np.imag(kspace)), -1)
kspace = torch.tensor(kspace)
print(kspace.shape)

moco = MocoSimLayer(maxTrans = 10, maxRot = 10, N_tp_Max = 16, half_width_no_motion = 12, noMotionFrac = 0.5)
und = UndersampleLayer()
for _ in range(10):
    kspace_sim = moco.forward(kspace)
    kspace_und, _ = und(kspace_sim)
    print(kspace.shape, kspace_sim.shape)
    img_ref = ks2img(kspace)
    img_ms = ks2img(kspace_sim)
    img_ms_und = ks2img(kspace_und)
    
    x = torch.tensor(np.expand_dims(img_ref, 0))
    x = kornia.geometry.transform.rotate(x, torch.tensor([45.]), center=None, mode='nearest', padding_mode='zeros', align_corners=True)
    x = np.squeeze(x)

    plt.subplot(1,4, 1)
    plt.imshow(img_ref, cmap='gray')
    plt.subplot(1,4, 2)
    plt.imshow(img_ms, cmap='gray')
    plt.subplot(1,4, 3)
    plt.imshow(img_ms_und, cmap='gray')
    plt.subplot(1,4, 4)
    plt.imshow(x, cmap='gray')
    plt.show()