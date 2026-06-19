#!/usr/bin/env python
#
# Do basecalling on batches of images.
# used for hyb

import argparse
import logging
import math
import os
import re
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
import joblib

import matplotlib.pylab as plt
import numpy as np

from skimage import color
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import extrema, binary_dilation
from skimage.util import img_as_float

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

#from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def basecall_hyb_ski( infiles, outfiles, stage=None, cp=None):
    '''
    take in all tiles for 1 cycle           
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'basecall-hyb'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    position_regex = cp.get(stage, 'position_regex')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, file_label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')

    # Stage-specific tool params
    all_genes_ch=cp.getint(stage, 'all_genes_ch')    
    thresh_str = cp.get( stage,'thresh')    
    relaxed = cp.getboolean( stage, 'relaxed')
    no_deconv = cp.getboolean( stage, 'no_deconv')
    filter_overlap = cp.getint( stage, 'filter_overlap')
    num_c = cp.getint( stage, 'num_c')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    
    # Parameters that need evaluation
    prominence_str = cp.get( stage, 'prominence')
    logging.debug(f'params. thresh_str={thresh_str} prominence_str={prominence_str} evaluating... ')
    prominence = eval( prominence_str ) 
    thresh = eval( thresh_str )
    logging.debug(f'all_genes_ch={all_genes_ch} thresh={thresh} prominence={prominence}')
          
    # Basecall loop.
    # Cycle list, contains 1 or more positions. 

    lroi_x_all=[]
    lroi_y_all=[]
    id_t_all=[]
    sig_t_all=[]

    for infile in infiles:
        (dirpath, base ) = os.path.split(infile)
        m = re.search(position_regex, base)
        if m is not None:
            pos_id = m.group(1)
        else:
            logging.error(f'Unable to extract position index from file base: {base}')
            sys.exit(2)
        logging.info(f'handling pos_id={pos_id}')

        #hyb_raw=tfl.imread(os.path.join(hybseq[0]), key=range(0,num_c,1))
        readchannels = list(range(0,num_c))
        hyb_raw=read_image(infile, channels=readchannels)
        # simply renaming hyb_raw, hyb_raw name not used again. 
        hyb_2=hyb_raw
        # zero-ing all-genes channel 3 (index 2)
        hyb_2[all_genes_ch,:,:] = 0
        [lroi_x_ind, lroi_y_ind, id_t_ind, sig_t_ind] = basecall_hyb_ski_single( infile,
                                                                                 outdir=outdir,
                                                                                 num_c=num_c, 
                                                                                 all_genes_ch=all_genes_ch, 
                                                                                 hyb_2=hyb_2,
                                                                                 thresh=thresh,
                                                                                 prominence=prominence)
        logging.debug(f'got result: lroi_x_ind={lroi_x_ind}, lroi_y_ind={lroi_y_ind}, id_t_ind={id_t_ind}, sig_t_ind={sig_t_ind} ')

        lroi_x_all.append([np.concatenate(lroi_x_ind) if any(len(x) for x in lroi_x_ind) else []])
        lroi_y_all.append([np.concatenate(lroi_y_ind) if any(len(x) for x in lroi_y_ind) else []])
        id_t_all.append([np.concatenate(id_t_ind) if any(len(x) for x in id_t_ind) else []])
        sig_t_all.append([np.concatenate(sig_t_ind) if any(len(x) for x in sig_t_ind) else []])

    data_dict = {"lroi_x":lroi_x_all, 
                 "lroi_y":lroi_y_all, 
                 "gene_id":id_t_all, 
                 "signal":sig_t_all}

    logging.info(f'Writing results to {outfile}')
    joblib.dump(data_dict, outfile)


# channel index -> gene index, from codebook_channels (1-based 1,2,4 -> 0-based 0,1,3).
# In MATLAB this is codes(:,m)==n (codebook lookup); the all-genes/registration
# channel (all_genes_ch) has no gene and is skipped.
CH_TO_GENE = {0: 0, 1: 1, 3: 2}


def basecall_hyb_ski_single(infile,
                            outdir,
                            num_c,
                            all_genes_ch,
                            hyb_2,
                            thresh,
                            prominence
                            ):
    '''
    Port of MATLAB mmbasecallhyb_multi (single-cycle). Per gene channel n:
      peaks = imregionalmax(imreconstruct(max(a-hybthresh,0), a))  ==  h_maxima(a, prominence[n])
      take the BRIGHTEST pixel of each maximal region (MATLAB max(a(component)));
      keep peaks with intensity > bgn (== thresh[n]);
      assign the gene for channel n (codebook), NOT argmax over channels.
    NOTE prominence[n] == MATLAB hybthresh (peak detection), thresh[n] == MATLAB
    bgn (final intensity filter). h_maxima is applied to the FULL channel (not a
    pre-masked image), matching MATLAB.
    '''
    lroi_x = []
    lroi_y = []
    id_t = []
    sig_t = []
    mask = np.zeros_like(hyb_2)
    for n in range(num_c):
        if n == all_genes_ch:
            continue
        a = np.asarray(hyb_2[n, :, :], dtype=np.float64)
        a_max = extrema.h_maxima(a, prominence[n])         # == imreconstruct + imregionalmax
        label_peaks = label(a_max)
        regions = regionprops(label_peaks, a)
        mask[n, :, :] = uint16m(binary_dilation(a_max))

        rx, ry, rid, rsig = [], [], [], []
        for peak in regions:
            coords = peak.coords                            # (row, col) pixels in maximal region
            vals = a[coords[:, 0], coords[:, 1]]
            bi = int(np.argmax(vals))                       # brightest pixel (MATLAB max(a(component)))
            prow, pcol = int(coords[bi, 0]), int(coords[bi, 1])
            sig = a[prow, pcol]
            if sig > thresh[n]:                             # MATLAB: a(peak) > bgn(n)
                ry.append(prow)                             # lroi_x = row, lroi_y = col
                rx.append(pcol)                             # (matches geneseq merge + aggregate mask[row,col])
                rid.append(CH_TO_GENE[n])
                rsig.append(sig)
        lroi_x.append(ry)
        lroi_y.append(rx)
        id_t.append(rid)
        sig_t.append(rsig)

    (dirpath, base ) = os.path.split(infile)
    (base,ext) = os.path.splitext(base)
    ext = ext[1:]
    of = os.path.join( outdir, f'{base}.mask_hyb.{ext}' )
    logging.debug(f'Writing mask to {of}')
    write_image(of, mask)

    gene_map = np.zeros(hyb_2.shape[1:], dtype=np.uint8)
    for ch_idx in range(len(lroi_x)):
        for k in range(len(lroi_x[ch_idx])):
            r = int(round(lroi_x[ch_idx][k]))
            c = int(round(lroi_y[ch_idx][k]))
            gene_map[r, c] = id_t[ch_idx][k] + 1  # +1 so background stays 0

    of = os.path.join( outdir, f'{base}.basecall_map_hyb.{ext}' )
    logging.debug(f'Writing basecall map to {of}')
    write_image(of, gene_map)
    return(lroi_x, lroi_y, id_t, sig_t)

if __name__ == '__main__':
    FORMAT='%(asctime)s (UTC) [ %(levelname)s ] %(filename)s:%(lineno)d %(name)s.%(funcName)s(): %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-d', '--debug', 
                        action="store_true", 
                        dest='debug', 
                        help='debug logging')

    parser.add_argument('-v', '--verbose', 
                        action="store_true", 
                        dest='verbose', 
                        help='verbose logging')

    parser.add_argument('-c','--config', 
                        metavar='config',
                        required=False,
                        default=os.path.expanduser('~/git/barseq-processing/etc/barseq.conf'),
                        type=str, 
                        help='config file.')
    
    parser.add_argument('-s','--stage', 
                    metavar='stage',
                    default=None, 
                    type=str, 
                    help='label for this stage config')

    parser.add_argument('-t','--template', 
                    metavar='template',
                    default=None,
                    required=False, 
                    type=str, 
                    help='label for this stage config')
    
    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='outfile. ')
       
    args= parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        loglevel = 'debug'
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)   
        loglevel = 'info'
    
    cp = ConfigParser()
    cp.read(args.config)
    cdict = format_config(cp)
    logging.debug(f'Running with config={args.config}:\n{cdict}')
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    basecall_hyb_ski( infiles=args.infiles, 
                       outfiles=args.outfiles,
                       stage=args.stage,  
                       cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

