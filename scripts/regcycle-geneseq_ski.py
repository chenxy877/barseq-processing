#!/usr/bin/env python
#
#
import argparse
import logging
import os
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np
import skimage as ski
from skimage.util import view_as_blocks
from scipy.signal import convolve2d , correlate2d, fftconvolve

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)
from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def regcycle_ski(infiles, outfiles, template=None, stage=None, cp=None ):
    '''
    
    @arg infiles    tiles across cycles
    @arg outdir     TOP-LEVEL out directory
    @arg template   optional file to use as template against infiles, 
                    otherwise register to first. 
    @arg cp         ConfigParser object
    @arg stage      stage label in cp
    
    '''
    if cp is None:
        cp = get_default_config()
    
    if stage is None:
        stage = 'regcycle'
    
    subsample_rate = int(cp.get(stage,'subsample_rate'))
    resize_factor = int(cp.get(stage,'resize_factor'))
    block_size = int(cp.get(stage,'block_size'))
    num_channels = int(cp.get(stage,'num_channels'))
    # MATLAB alignseq_maxthresh: geneseq uses intensity_max_thresh=0 (pure
    # top-hat + xcorr, no soma masking); disk top-hat radius is 10 (resized px).
    intensity_max_thresh = int(cp.get(stage, 'intensity_max_thresh', fallback='0'))
    disk_radius = int(cp.get(stage, 'reg_disk_radius', fallback='10'))
    logging.info(f' stage={stage} template={template}')
    logging.debug(f'num_channels={num_channels} block_size={block_size} '
                  f'resize_factor={resize_factor} subsample_rate={subsample_rate} '
                  f'intensity_max_thresh={intensity_max_thresh}')

    if template is None:
        fixed_file = infiles[0]
    else:
        fixed_file = template
    logging.debug(f'fixed_file = {fixed_file}')

    fixed = read_image( fixed_file )
    # MATLAB sums the first num_channels (seq) channels on the ORIGINAL double
    # scale (no [0,1] normalization, no uint8 quantization).
    fixed_sum = np.sum(np.asarray(fixed[:num_channels], dtype=np.float64), axis=0)

    for i, infile in enumerate( infiles ):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        moving = read_image( infile )
        moving_sum = np.sum(np.asarray(moving[:num_channels], dtype=np.float64), axis=0)

        # Block-wise soma-avoiding FFT cross-correlation (alignseq_maxthresh).
        xoff, yoff = block_xcorr_translation(fixed_sum, moving_sum,
                                             block_size=block_size,
                                             resize_factor=resize_factor,
                                             subsample_rate=subsample_rate,
                                             intensity_max_thresh=intensity_max_thresh,
                                             disk_radius=disk_radius)
        logging.debug(f'transform for {infile} -> {fixed_file}: xoff={xoff:.3f} yoff={yoff:.3f}')

        # MATLAB applies imtranslate(im, [xoff,yoff]); skimage.warp uses the
        # inverse map, so translate by the negative to reproduce imtranslate.
        tform = ski.transform.SimilarityTransform(translation=[-xoff, -yoff])
        moving_aligned = np.zeros_like(moving)
        for ch in range(moving.shape[0]):
            moving_aligned[ch,:,:] = ski.transform.warp(np.squeeze(moving[ch,:,:]),
                                                        tform,
                                                        preserve_range=True,
                                                        output_shape=(moving.shape[1], moving.shape[2]))
        moving_aligned = uint16m(moving_aligned)
        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, moving_aligned)
        logging.debug(f'done writing {outfile}')


    
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

    (outdir, file) = os.path.split(args.outfiles[0])
          
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    regcycle_ski( infiles=args.infiles,  
                  outfiles=args.outfiles,
                  template=args.template, 
                  stage=args.stage, 
                  cp=cp )
    
    logging.info(f'done processing output to {outdir}')