#
#  Initial library to abstract out image handling. 
#  May allow substitution of file formats and tiff/imageio versions.
#  Will only work if version compatibility with the various tool environments is tolerant. 
#
#
#  Our standard is CHANNEL-FIRST   ( 5, 3200, 3200 )
#  channel, row, column    CXY
#
#
import logging
import os

import tifffile
from tifffile import imread, imwrite, TiffFile, TiffWriter

import imageio.v3 as iio
import numpy as np

# These back the registration / morphology helpers below, which only ever run in
# the `barseq` (ski/np/py/cv2) env. imageutils is imported by EVERY stage script
# (including denoise=n2v_bsp and bardensr=bardensr_bsp, which lack cv2/skimage),
# so guard the imports — importing this module must never fail on a missing dep.
try:
    import scipy.ndimage as ndi
except ImportError:
    ndi = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import skimage as ski
except ImportError:
    ski = None
try:
    from scipy.signal import fftconvolve
except ImportError:
    fftconvolve = None


def _disk_tophat(a, radius):
    '''Flat disk top-hat: a - opening(a, disk(radius)), matching MATLAB
    a - imopen(a, strel('disk', radius)). Uses cv2 (fast) with an elliptical
    (disk) flat SE.'''
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    a32 = np.asarray(a, dtype=np.float32)
    opened = cv2.morphologyEx(a32, cv2.MORPH_OPEN, k)
    return (a32 - opened).astype(np.float64)


def xcorr2_avoidsoma(a, b, thresh, radius=10):
    '''
    Port of MATLAB my_xcorr2_avoidsoma_fft (used by alignseq_maxthresh /
    align_local). 'same'-mode FFT cross-correlation of a and b after a flat
    disk-radius top-hat (high-pass). For thresh > 0 it additionally masks
    bright ("soma") pixels and reweights the correlation by the fraction of
    below-threshold overlap (nan_count/total_count). thresh == 0 (geneseq) is
    just top-hat + correlation.
    '''
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    bflip = lambda x: x[::-1, ::-1]
    if thresh > 0:
        nan_count = fftconvolve((a <= thresh).astype(np.float64),
                                bflip((b <= thresh).astype(np.float64)), mode='same')
        total_count = fftconvolve(np.ones_like(a), bflip(np.ones_like(b)), mode='same')
        a_mask = a > thresh
        b_mask = b > thresh
        a = _disk_tophat(a, radius)
        b = _disk_tophat(b, radius)
        a[a_mask] = 0
        b[b_mask] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            c = fftconvolve(a, bflip(b), mode='same') * nan_count / total_count
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        a = _disk_tophat(a, radius)
        b = _disk_tophat(b, radius)
        c = fftconvolve(a, bflip(b), mode='same')
    return c


def block_xcorr_translation(fixed_sum, moving_sum, block_size, resize_factor,
                            subsample_rate, intensity_max_thresh, disk_radius=10):
    '''
    Port of MATLAB alignseq_maxthresh / align_local: estimate a single
    integer-ish translation (xoff, yoff) aligning moving_sum to fixed_sum.

    fixed_sum, moving_sum : 2D float, ORIGINAL intensity scale (NOT normalized,
        NOT uint8-quantized -- matches MATLAB which keeps the summed channels as
        double).
    Steps: crop to a whole number of block_size blocks; split into blocks;
    keep the brightest 1/subsample_rate blocks (by fixed-image intensity);
    resize each kept block by resize_factor; soma-avoiding FFT cross-correlate;
    accumulate; take the peak nearest the origin. Offsets are in original-image
    pixels.
    '''
    fs = np.asarray(fixed_sum, dtype=np.float64)
    ms = np.asarray(moving_sum, dtype=np.float64)
    bn1 = fs.shape[0] // block_size
    bn2 = fs.shape[1] // block_size
    fs = fs[:bn1 * block_size, :bn2 * block_size]
    ms = ms[:bn1 * block_size, :bn2 * block_size]

    f_blocks, m_blocks = [], []
    for i in range(bn1):
        for j in range(bn2):
            sl = (slice(i * block_size, (i + 1) * block_size),
                  slice(j * block_size, (j + 1) * block_size))
            f_blocks.append(fs[sl])
            m_blocks.append(ms[sl])

    fsum = np.array([blk.sum() for blk in f_blocks])
    order = np.argsort(fsum)[::-1]                 # brightest first
    ntop = int(round(len(f_blocks) / subsample_rate))

    csize = block_size * resize_factor
    c = np.zeros((csize, csize), dtype=np.float64)
    for k in range(ntop):
        idx = order[k]
        if f_blocks[idx].max() > 0:
            # MATLAB imresize default is bicubic (order=3); upsampling does not antialias.
            fa = ski.transform.resize(f_blocks[idx], (csize, csize), order=3,
                                       mode='edge', preserve_range=True, anti_aliasing=False)
            mb = ski.transform.resize(m_blocks[idx], (csize, csize), order=3,
                                       mode='edge', preserve_range=True, anti_aliasing=False)
            c = c + xcorr2_avoidsoma(fa, mb, intensity_max_thresh, disk_radius)

    # peak nearest the origin (MATLAB: min(|xoff|+|yoff|) over all argmax ties).
    # For 'same'-mode correlation the zero-lag is at index N//2 (== block*rf/2);
    # scipy's 0-indexed peak and MATLAB's 1-indexed peak coincide there.
    peaks = np.argwhere(c == c.max())
    half = block_size * resize_factor / 2.0
    offs = [((p[1] - half) / resize_factor, (p[0] - half) / resize_factor)
            for p in peaks]
    xoff, yoff = min(offs, key=lambda xy: abs(xy[0]) + abs(xy[1]))
    return xoff, yoff


def expand_labels_matlab(mask, radius):
    '''
    Replicate MATLAB import_cellpose mask dilation:

        for n = 1:radius
            maski1 = imdilate(maski_dilated, ones(3));
            maski_dilated(maski_dilated==0) = maski1(maski_dilated==0);
        end

    i.e. iteratively (radius times) dilate the label image with a 3x3 box
    (8-connected) and fill ONLY background (0) pixels with the dilated label.
    Existing labels are never overwritten; where two labels reach a background
    pixel in the same iteration the higher label value wins (max filter).

    This is NOT the same as skimage.segmentation.expand_labels (Euclidean
    distance, nearest-label) which the Python code used previously.
    '''
    md = np.asarray(mask).copy()
    for _ in range(int(radius)):
        dil = ndi.maximum_filter(md, size=3, mode='constant', cval=0)
        bg = md == 0
        md[bg] = dil[bg]
    return md


def label_median_centroids(labelimg):
    '''
    Per-label median position, matching MATLAB import_cellpose:
        [y,x] = ind2sub(size, find(imcell==label));
        cellpos(n,:) = [median(x), median(y)];
    Returns (labels, cent_x, cent_y) where cent_x is median column and cent_y
    is median row. Background label 0 is excluded.
    '''
    labelimg = np.asarray(labelimg)
    labels = np.unique(labelimg)
    labels = labels[labels != 0]
    cent_x = np.empty(len(labels), dtype=np.float64)
    cent_y = np.empty(len(labels), dtype=np.float64)
    for i, lab in enumerate(labels):
        ys, xs = np.where(labelimg == lab)
        cent_x[i] = np.median(xs)
        cent_y[i] = np.median(ys)
    return labels, cent_x, cent_y


def matlab_ball_structure(radius):
    '''
    Replicate MATLAB strel('ball', R, R, 0) -- the non-flat ("true ball")
    grayscale structuring element used by run_barseq for background subtraction.

    For strel('ball', R, H) the height profile is
        height(dx,dy) = (H/R) * sqrt(R^2 - dx^2 - dy^2)   for dx^2+dy^2 <= R^2
    run_barseq uses H == R, so height = sqrt(R^2 - dx^2 - dy^2).

    Returns (footprint_bool, height_float) matching MATLAB getnhood/getheight.

    NOTE: production MATLAB calls strel('ball', R, R) which defaults to N=8, a
    line-segment *approximation* of the ball (for speed). The true ball (N=0)
    differs from the N=8 approximation by a few intensity units. We replicate the
    true ball, which is the intended algorithm; see scratch/exp_ball.* for the
    validation (true-ball Python matches MATLAB N=0 to <1 count everywhere).
    '''
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = xx * xx + yy * yy
    footprint = d2 <= r * r
    height = np.zeros(d2.shape, dtype=np.float64)
    height[footprint] = np.sqrt(r * r - d2[footprint])
    return footprint, height


def ball_opening(image2d, radius):
    '''
    Grayscale morphological opening of a 2D image with a non-flat ball SE,
    matching MATLAB imopen(im, strel('ball', radius, radius)).
    Returns the opened (background) image as float64.
    '''
    footprint, height = matlab_ball_structure(radius)
    f = np.asarray(image2d, dtype=np.float64)
    return ndi.grey_opening(f, footprint=footprint, structure=height)


def ball_tophat(image2d, radius, exact_max_radius=24):
    '''
    Background top-hat: im - imopen(im, strel('ball', r, r)), matching the
    MATLAB fixbleed background subtraction. MATLAB rounds the opening to the
    image integer type before subtracting and clamps negatives to 0
    (opening is anti-extensive so the result is non-negative).

    For small radii (<= exact_max_radius, e.g. geneseq r=6) this uses the exact
    true-ball grey opening (matches MATLAB N=0 to <1 count). For large radii
    (e.g. hyb r=100) the exact 2D grey opening is far too slow (~11 min/channel
    at r=100), so the ball background is APPROXIMATED by opening a downsampled
    image and upsampling the background.

    FLAG: the large-radius path is an approximation of MATLAB's ball opening
    (which itself uses the N=8 line approximation for large balls). Replace with
    an exact N=8 line-decomposition port if hyb background precision matters.
    Returns float64.
    '''
    f = np.asarray(image2d, dtype=np.float64)
    r = int(radius)
    if r <= exact_max_radius:
        opened = np.rint(ball_opening(f, r))
    else:
        # Large radius (e.g. hyb r=100): the exact grey ball opening is ~11 min/
        # channel. At large R the (very tall, height==R) ball opening is well
        # approximated by a FLAT disk opening (validated vs MATLAB ball-100:
        # corr ~0.99, mean within ~1.5 counts; see scratch/exp_ball100.*), and
        # cv2 makes it tractable (~20s/channel). FLAG: flat-disk approximation of
        # the tall ball -- replace with an exact N=8 line decomposition if hyb
        # background precision proves to matter.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(f.astype(np.float32), cv2.MORPH_OPEN, k).astype(np.float64)
        opened = np.rint(opened)
    return np.clip(f - opened, 0, None)


def channel_names_index_map(select_channels, image_channels):
    '''

    Given ordered channel name list and list of desired channels, 
    return index list to pull from input image. 

    image_channels = ['GFP', 'YFP', 'TxRed', 'Cy5', 'DAPI', 'BF']
    select_channels = ['GFP', 'YFP', 'TxRed', 'Cy5']
    ->
    [0,1,2,3]

    select_channels = ['TxRed', 'GFP', 'YFP', 'Cy5']
    -> [2,0,1,3]
    '''
    idx_list = []
    for selname in select_channels:
        try:
            ch_idx = image_channels.index(selname)
            idx_list.append(ch_idx)
        except ValueError as ve:
            logging.error('Channel name {selname} not in image_channels={image_channels}. Check config. ')
            raise
    return idx_list


def read_image(infile, channels=None):
    '''
    BARseq standard image interface. 
    Intended to abstract out underlying formats and libraries. 
    image is numpy.ndarray, where shape = (channel, y|height , x|width )
    channels is list of integer *indexes*, starting at 0. 

    Caller is responsible for converting Channel numbers to indexes (i.e. C - 1)
    '''
    np_array = iio.imread(infile)
    #logging.debug(f'read image shape={np_array.shape} from {infile}')
    if channels is not None:
        if len(channels) == 1:
            new_array = np_array[channels[0]]
        else:
            new_array = np.ndarray( ( len(channels), np_array.shape[1], np_array.shape[2] ) ) 
            for i, channel in enumerate( channels ):
                new_array[i] = np_array[channel]
                logging.debug(f'reading channel idx={channel} shape={np_array.shape}')
        np_array = new_array
    return np_array

def write_image(outfile, np_array, photometric='minisblack'):
    '''
    BARseq standard image interface. 
    Intended to abstract out underlying formats and libraries. 
    image is numpy.ndarray, where shape = (channel, y|height , x|width )
    Assuming tif plugin, photometric = [ 'minisblack' | 'miniswhite'| 'rgb' ]
    '''
    # iio.v3 syntax. pass args as-is to underlying plugin
    iio.imwrite( outfile, np_array, photometric=photometric )
    
    # iio v2 syntax explicitly create plugin_kwargs dict. 
    #iio.imwrite( outfile, np_array, plugin_kwargs={"photometric": photometric})
    logging.debug(f'wrote image shape={np_array.shape} photometric={photometric} to {outfile}')
 
 
#   
# Ashlar-specific image handling.
#
    
def write_mosaic(mosaic, outfile):
    #for ci, channel in enumerate(self.mosaic.channels):
    #channel = 0
    if self.verbose:
        logging.info(f"Assembling channel {channel}:")
    img = self.mosaic.assemble_channel(channel)
    img = uint16m(img)
    images.append(img)
    img = None
    #(dirpath, base, ext) = split_path(self.outpath)
    #outfile = os.path.join(dirpath, f'{base}.{channel}.{ext}')
    logging.debug(f'Added channel {channel} to image list.')

    fullimage = np.dstack(images)
    logging.debug(f'dstack() -> {fullimage.shape}')
    # produces e.g. shape = ( 3200,3200,5)
    fullimage = np.rollaxis(fullimage, -1)
    logging.debug(f'rollaxis() -> {fullimage.shape}')
    # produces e.g. shape = ( 5, 3200, 3200)  
    

#    
#  Bardensr specific image handling.
#
def bd_read_images(infiles, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    might be useful elsewhere...
    
    '''
    I = []
    for infile in infiles:
        for j in range(C):
            I.append( np.expand_dims( read_image( infile, channels=[j]), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[ :, :, round(ny*cropf):round(ny*(1-cropf)), round(nx*cropf):round(nx*(1-cropf)) ]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:, :, trim:-trim, trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    logging.debug(f'created image stack dimensions={I.shape}')
    return I


def bd_read_image_set(infiles, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    assumes input is set of multiple cycles images for single tile

    '''
    I = []
    #for i in range(1, R+1):
    for infile in infiles:
        # logging.debug(f'reading {infile}')
        for j in range(C):
            I.append( np.expand_dims( read_image( infile, channels=[j]), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[ :, :, round(ny*cropf):round(ny*(1-cropf)), round(nx*cropf):round(nx*(1-cropf)) ]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:, :, trim:-trim, trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    return I

def bd_read_image_single(infile, R, C, trim=None, cropf=None ):
    '''
    specialized image handling for bardensr with crop/trim 
    might be useful elsewhere...
    
    '''
    I = []
    for i in range(1, R+1):
        for j in range(C):
            I.append( np.expand_dims( read_image( infile, channels=[j]), axis=0))
    I=np.array(I)
    if cropf is not None:
        logging.debug(f'cropping image by: {cropf}')
        nx = np.size(I,3)
        ny = np.size(I,2)
        I = I[ :, :, round(ny*cropf):round(ny*(1-cropf)), round(nx*cropf):round(nx*(1-cropf)) ]
    elif trim is not None:
        logging.debug(f'trimming image by: {trim}')
        I = I[:, :, trim:-trim, trim:-trim]
    else:
        logging.debug(f'no mods requests. returning all channels.')
    return I