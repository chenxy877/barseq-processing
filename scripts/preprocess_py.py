#!/usr/bin/env python
#
#
# Combination background, regchannels, and bleedthrough. 
#
#
import argparse
import logging
import os
import sys
import datetime as dt

import cv2
import numpy as np
import skimage as ski
import tifffile as tf

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def background_ball_single(image, radius, num_c=4):
    '''
    Background subtraction matching MATLAB fixbleed:
    per (seq) channel, im - imopen(im, strel('ball', radius, radius)).

    Uses a grayscale (non-flat) BALL structuring element via
    imageutils.ball_tophat -- NOT a flat cv2 disk, which is what the original
    Python used. Extra channels (e.g. BF) beyond num_c are preserved unchanged.
    '''
    I = image
    I_filtered = np.zeros(I.shape, dtype=np.float64)
    for i in range(num_c):
        I_filtered[i, :, :] = ball_tophat(I[i, :, :], radius)
    # preserve extra channel(s) unchanged
    I_filtered[num_c:, :, :] = I[num_c:, :, :].astype(np.float64)
    return uint16m(I_filtered)

def regchannels_ski_single(image, channel_shift, is_affine=False):
    '''
    
    '''
    n_channels = len(channel_shift)
    I=image.copy()
    Ishifted=np.zeros_like(I)
    # Save extra channel(s)
    I_rem=I[n_channels:,:,:]
    I=I[0:n_channels,:,:]
    for i in range(channel_shift.shape[0]):
        if is_affine:
            # refine this later on-ng
            tform=channel_shift[i] 
        else:
            # remember this takes -shifts-ng  
            tform=ski.transform.SimilarityTransform(translation = -channel_shift[i,:])              
        It=ski.transform.warp(np.squeeze(I[i,:,:]), 
                                    tform, 
                                    preserve_range=True, 
                                    output_shape=(I.shape[1],I.shape[2]))
        Ishifted[i,:,:]=np.expand_dims(It,0)
    # Put back extra channel(s)
    Ishifted[n_channels:,:,:]=I_rem    
    Ishifted=uint16m(Ishifted)
    return Ishifted

def bleedthrough_np_single(image, chprofile):

    n_channels = len(chprofile)
    I=image.copy()
    Icorrected=np.zeros_like(I)
    Ishifted2 = np.float64( I[0:n_channels,:,:] )
    I_rem=I[n_channels:,:,:]
    A = np.transpose(chprofile)
    B = np.reshape( Ishifted2 , (n_channels, -1), order='F')
    I_solved = np.linalg.solve( A, B ) 
    Icorrected=np.reshape( I_solved , 
                            ( n_channels, Ishifted2.shape[1], Ishifted2.shape[2]),
                            order='F')       
    Icorrected=uint16m(Icorrected)
    Icorrected=np.append(Icorrected, I_rem, axis=0)
    return Icorrected



def preprocess_py( infiles, outfiles, stage=None, cp=None):
    '''
    Perform background, regchannels, and bleedthrough correction.
    
    hyb denoised    = 6 channels
    geneseq denoised = 5 channels. 
    [geneseq]
    channels=G,T,A,C,BF

    [hyb]
    channels=GFP,YFP,TxRed,Cy5,DAPI,BF
       
    num_initial_c=5,
    num_later_c=4
    num_c=4

    num_c_hyb=5

    geneseq: 
    hyb: num_c=num_c_hyb

    preprocess():
    process_geneseq_cycle(  num_initial_c=num_initial_c,
                            num_later_c=num_later_c,
                            num_c=num_c)
    process_hyb_cycle(      num_c=num_c_hyb,
                            is_affine=is_affine)
    
    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'preprocess-geneseq'

    # Get parameters for all steps.
    mode = get_config_list(cp, stage, 'modes')
    mode = mode[0]
    # Background ball radius matches MATLAB ball_radius (geneseq=6, hyb=100),
    # read per-stage. (MATLAB applies a grayscale 'ball' SE, see ball_tophat.)
    radius = int(cp.get(stage, 'background_radius', fallback='0'))
    output_dtype = cp.get( stage,'output_dtype')
    logging.debug(f'output_dtype={output_dtype} radius = {radius} ')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    microscope_profile = cp.get('experiment','microscope_profile')
    chshift_file = cp.get(microscope_profile,f'channel_shift_{mode}')
    chshift_path = os.path.join(resource_dir, chshift_file)
    is_affine = cp.getboolean(stage,'is_affine')
    logging.debug(f'chshift_path = {chshift_path} is_affine={is_affine}')
    chshift = load_df(chshift_path, as_array=True)
    n_shift_channels = len(chshift)
    logging.debug(f'loaded channel shift. len={n_shift_channels} ')
    chprofile_file = cp.get(microscope_profile, f'channel_profile_{mode}')
    chprofile_path = os.path.join(resource_dir, chprofile_file)
    chprofile = load_df(chprofile_path, as_array=True)
    num_prof_channels = len(chprofile)
    logging.debug(f'chprofile_file={chprofile_file} num_channels={num_prof_channels}')

    for i, infile in enumerate(infiles):
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        logging.info(f'Handling {infile} -> {outfile}')                
        (dirpath, base, label, ext) = split_path(os.path.abspath(infile))

        I = read_image( infile)

        # MATLAB fixbleed order: (1) shift channels, (2) background, (3) bleedthrough.
        # 1. Channel registration (chromatic shift) -- applied FIRST in MATLAB.
        logging.debug(f'Do regchannels (shift)...')
        Ishifted = regchannels_ski_single(image=I, channel_shift=chshift)
        logging.debug(f'Done with regchannels. mode={mode} n_channnels={len(Ishifted)}')

        # 2. Background subtraction (grayscale ball top-hat).
        if radius > 0:
            Ibacksub = background_ball_single(Ishifted, radius)
        else:
            Ibacksub = Ishifted
        logging.debug(f'Done background. mode={mode} n_channnels={len(Ibacksub)}')

        # 3. Bleedthrough correction.
        logging.debug(f'Do bleedthrough...')
        Icorrected = bleedthrough_np_single(Ibacksub, chprofile)
        logging.debug(f'Done bleedthrough. mode={mode} n_channnels={len(Icorrected)}')
        logging.debug(f'done processing {base}.{ext} ')
        logging.info(f'writing to {outfile}')
        write_image(outfile, Icorrected)
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

    preprocess_py( infiles=args.infiles,
                     outfiles=args.outfiles,
                     stage=args.stage,
                     cp=cp )
    (outdir, file) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')