import nibabel as nb
import numpy as np
import argparse
from scipy import ndimage
from PIL import Image
import sys
from matplotlib import pyplot as plt


def qc_slice(image_path: str,
             save_path: str = None,
             mask_path: str = None,
             mask_template: str = None,
             nslices: int = 1):
    '''
    Extracts and plots slices in each plane from the specified 3D image. Optionally overlays a mask on the image. Slices
    are centered on the image midpoint.

    Parameters
    ----------
    image_path : str
        Filepath to the image to slice.
    save_path : str
        Optional. If not set, will plot image via matplotlib. Path where to save the resulting image.
    mask_path : str
        Optional. Mask to overlay. Default: None.
    mask_template : str
        Optional. Template mask to overlay. Default: None.
    nslices : int
        Optional. The number of slices to take.

    Returns
    -------
    None
    '''

    image = nb.load(image_path)
    if(mask_path is not None):
        mask = nb.load(mask_path).get_fdata()
    if(save_path is not None):
        if(not save_path.endswith('.png')):
            save_path += '.png'
    if(mask_template is not None):
        template = nb.load(mask_template).get_fdata()

    imshape = image.shape
    slice_ind = [
        np.linspace(
            0,
            imshape[i],
            nslices +
            2,
            dtype=int)[
            1:-
            1] for i in range(3)]
    image = image.get_fdata()
    immax = np.max(image)

    full_img = np.zeros(
        (imshape[2] * 2 + imshape[1], np.max([imshape[1], imshape[0]]) * nslices, 3))
    prev_lim = 0
    for i in range(3):
        slices = [slice(None) for _ in range(3)]
        if(i == 0):
            slice_im = np.zeros((imshape[2], imshape[1] * nslices, 3))
            mlt = imshape[1]
        elif(i == 1):
            slice_im = np.zeros((imshape[2], imshape[0] * nslices, 3))
            mlt = imshape[0]
        else:
            slice_im = np.zeros((imshape[1], imshape[0] * nslices, 3))
            mlt = imshape[0]

        for j in range(nslices):
            start_ind = j * mlt
            slices[i] = slice_ind[i][j]
            slice_im[:,
                     start_ind: start_ind + mlt,
                     0] = ndimage.rotate(image[slices[0],
                                               slices[1],
                                               slices[2]],
                                         90)
            slice_im[:, start_ind: start_ind + mlt,
                     1] = slice_im[:, start_ind: start_ind + mlt, 0]
            slice_im[:, start_ind: start_ind + mlt,
                     2] = slice_im[:, start_ind: start_ind + mlt, 0]
            if(mask_path is not None):
                slice_im[:,
                         start_ind: start_ind + mlt,
                         0] += ndimage.rotate(mask[slices[0],
                                                   slices[1],
                                                   slices[2]],
                                              90) / np.max(mask) * 0.2 * immax
            if(mask_template is not None):
                slice_im[:,
                         start_ind: start_ind + mlt,
                         1] += ndimage.rotate(template[slices[0],
                                                       slices[1],
                                                       slices[2]],
                                              90) / np.max(template) * 0.2 * immax
        full_img[prev_lim:prev_lim + slice_im.shape[0],
                 :slice_im.shape[1], :] = slice_im / immax
        prev_lim += slice_im.shape[0]

    save_im = np.array((full_img - np.min(full_img)) *
                       255 / np.max(full_img), dtype=np.uint8)
    im = Image.fromarray(save_im)
    if(save_path is not None):
        im.save(save_path)
    else:
        plt.imshow(im)
    return


def main(args):
    parser = argparse.ArgumentParser(
        'Take N regularly-sampled slices from input image')
    parser.add_argument('image', help='Path of the image to slice.')
    parser.add_argument('save_path', help='Path where to save QC image')
    parser.add_argument(
        '--mask_path',
        default=None,
        help='Path to the mask file')
    parser.add_argument(
        '--mask_template',
        default=None,
        help='Path to the template mask file')
    parser.add_argument(
        '--nslices',
        default=1,
        type=int,
        help='The number of slices to take in each direction.')
    pargs = parser.parse_args(args)
    qc_slice(pargs.image, pargs.save_path, mask_path=pargs.mask_path,
             nslices=pargs.nslices, mask_template=pargs.mask_template)
    return


if __name__ == '__main__':
    main(sys.argv[1:])
