#!/usr/bin/env python
#
# Use Cellpose to segemnt cells. 
#
# current inputs. 
#         hyb:  5 channels. 
#             hyb.  channel 3 (all-genes)
#             hyb.  channel 5. DAPI
#         geneseq:
#             sum(all_channels) from either geneseq01 or all geneseq* 
#
import argparse
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser
from joblib import load, dump

import torch
import numpy as np

from cellpose import models, io
from cellpose.io import imread

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

def segment_cellpose( infiles, outfiles, stage=None, cp=None):
    '''
    take in infiles of same tile through multiple cycles, 
    create imagestack, 
    run cellpose
      
    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'segment'

    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    model_name = cp.get(stage, 'model_name')
    cell_diameter = cp.getint(stage, 'cell_diameter')
    use_gpu = torch.cuda.is_available()
    logging.info(f'running with model_name={model_name} cell_diam={cell_diameter} use_gpu={use_gpu}')

    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]} ')
    (dirpath, base, infile_label, ext) = split_path(os.path.abspath(infiles[0]))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    cellpose_input_stack = prepare_cellpose_input(infiles, outfiles )
    logging.debug(f'got cellpose input image shape={cellpose_input_stack.shape}')

    model = models.Cellpose( model_type = model_name,
                             gpu = use_gpu)
    # MATLAB Cellsegmentation-v065 uses channels=[1,2] (cytoplasm=1st channel,
    # nucleus=2nd channel) on an (H,W,2) array. cp_input_image here is (2,H,W)
    # with cyto at index 0, nucleus at index 1. channels=[1,2] = [cyto=1st,
    # nucleus=2nd]; the previous [[0,1]] meant cyto=grayscale which is wrong.
    # FLAG: verify channel_axis/order interaction with cellpose in the live test.
    channels = [1, 2]
    logging.info('running cellpose...')
    masks, flows, styles, diams = model.eval( cellpose_input_stack, 
                                              diameter=cell_diameter, 
                                              channels=channels )
    logging.debug(f'got masks. shape={masks.shape}')

    logging.info(f'writing to {outfile}')
    write_image(outfile, masks)
    logging.debug(f'done writing {outfile}')    


def prepare_cellpose_input(infiles, outfiles):
    '''
        cyto = hyb[0,1,2,3] + geneseq all channel all cycle composite, 
        nuclear = hyb[4].

        nuc_ch=5,
        num_chyb=5,
        num_cgene=4,
        other_channels = list(range(0,num_chyb))
    '''
    # We know arity is single, so we can grab the outfile 
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')    

    (odirpath, obase, olabel, oext) = split_path(os.path.abspath(outfile))
    (prefix, subdir) = os.path.split(odirpath)
    logging.debug(f'outdirpath={odirpath} obase={obase} olabel={olabel} subdir={subdir}')
    outfile = os.path.join( outdir, f'{obase}.cp_inp.tif' )
    logging.debug(f'preparing cellpose input to be written to {outfile}')

    hyb_image = read_image(infiles[0])
    cp_input_image = np.zeros( [2, hyb_image.shape[1], hyb_image.shape[2]] )
    gene_composite = np.zeros( [ hyb_image.shape[1],hyb_image.shape[2] ] )
    # MATLAB Cellsegmentation-v065: cytoplasm = hyb[0:3] + geneseq01[0:3], i.e.
    # the first 3 channels of ONLY the first geneseq cycle (not all 4 channels,
    # not all cycles); nuclear = hyb[4] (DAPI).
    if len(infiles) > 1:
        gene_image = read_image(infiles[1], channels=[0, 1, 2] )
        gene_composite = np.sum( gene_image, axis=0 )
    nuclear_image = hyb_image[4]
    cyto_image = np.sum( hyb_image[0:3], axis=0 ) + gene_composite
    cp_input_image[0,:,:]=uint16m(cyto_image)
    cp_input_image[1,:,:]=uint16m(nuclear_image)
    logging.debug(f'made cellpose input image. shape={cp_input_image.shape}')
    logging.debug(f'writing intermediate cellpose input to {outfile} ...')
    write_image(outfile, cp_input_image)
    logging.debug(f'returning intermediate image...')
    return cp_input_image


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

    segment_cellpose( infiles=args.infiles, 
                      outfiles=args.outfiles,
                      stage=args.stage,  
                      cp=cp )
    
    logging.info(f'done processing output to {args.outfiles[0]}')

