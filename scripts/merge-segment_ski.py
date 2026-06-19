#!/usr/bin/env python
#
# merges and dilates segmentation data for all tiles in a positioin.
#
#
import argparse
import joblib
import logging
import os
import re
import sys

import datetime as dt
from configparser import ConfigParser

import numpy as np

from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops_table


gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def merge_segment_ski( infiles, outfiles, stage=None, cp=None ):
    
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'merge-segment'

    # We know arity is single, so we can grab the single outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')
       
    # get params
    dilation_radius = cp.getint(stage,'dilation_radius')

    all_segmentation_dict={}
    for infile in infiles:
        (subdir, base, label, ext) = parse_rpath(infile)
        logging.debug(f'handling subdir={subdir} base={base} label={label} ext={ext}')
        tile_data={}
        [mask, mask_dil, cell_num, cent_x, cent_y] = handle_single_tile_segmentation(infile, dilation_radius)
        tile_data['original_labels']= mask
        tile_data['dilated_labels']= mask_dil
        tile_data['cell_num']=cell_num
        tile_data['cent_x']=cent_x
        tile_data['cent_y']=cent_y
        all_segmentation_dict[base]=tile_data
        #io.imsave(os.path.join(pth,'processed',folder,'aligned','dil_'+fname), mask_dil)
    logging.info(f'writing output to {outfile}')
    joblib.dump(all_segmentation_dict, outfile )
    logging.info(f'Done.')
    
    
def handle_single_tile_segmentation(infile, dilation_radius):
    logging.debug(f'handling infile {infile} dilation_radius = {dilation_radius}')
    mask = read_image( infile )
    # MATLAB import_cellpose: iterative imdilate(ones(3)) filling only
    # background, and per-cell MEDIAN centroid in [x=col, y=row] order
    # (skimage expand_labels + mean regionprops centroid was the prior, divergent code).
    mask_dil = expand_labels_matlab( mask, dilation_radius)
    cell_num, cent_x, cent_y = label_median_centroids(mask_dil)
    return mask,mask_dil,cell_num,cent_x,cent_y


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
                        help='File[s] to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    default=None, 
                    nargs ="+",
                    type=str,  
                    help='Output file[s]. ') 
       
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

    merge_segment_ski( infiles=args.infiles, 
                       outfiles=args.outfiles, 
                       cp=cp )
    (outdir, fname) = os.path.split(args.outfiles[0])
    logging.info(f'done processing output to {outdir}')