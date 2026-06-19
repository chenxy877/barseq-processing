#!/usr/bin/env python
#
# Script to use noise2void to do denoising on TIF images based on trained models. 
# Script takes TIF file list as input and places output to outdir
#
# 
# General filename scheme:  <filebase>.<pipelinestage>.tif
# Filenames altered as:
#       MAX_Pos1_000_000.tif  -> MAX_Pos1_000_000.denoised.tif  
#
# https://github.com/juglab/n2v
# https://csbdeep.bioimagecomputing.com/
# 
# https://imageio.readthedocs.io/en/v2.9.0/userapi.html
#    2.36.1 
#
import argparse
import logging
import os
import sys

# Determinism: TensorFlow's oneDNN custom ops reorder float accumulation and make
# n2v predictions non-reproducible run-to-run (e.g. hyb ch0 86.5 vs 97.1). Disabling
# oneDNN makes the denoise deterministic AND bit-match the MATLAB n2v output exactly,
# which removes the n2v-noise that was leaking into the bardensr threshold calibration
# and the hyb basecalling. Must be set BEFORE tensorflow is imported (via n2v below).
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

#from tensorflow import keras
from n2v.models import N2V
import numpy as np

def denoise_n2v( infiles, outfiles, stage=None, cp=None):
    '''
    
    def predict(self, img, axes, 
                    resizer=PadAndCropResizer(), 
                    n_tiles=None, 
                    tta=False):

    '''
    if cp is None:
        cp = get_default_config()
    if stage is None:
        stage = 'denoise'
            
    logging.info(f'handling stage={stage} n_infiles={len(infiles)} n_outfiles={len(outfiles)}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    basedir = os.path.join(resource_dir, 'n2vmodels')
    image_type = cp.get(stage,'image_type')
    logging.debug(f'this mode is image_type={image_type}')
    image_channels = cp.get(image_type, 'channels').split(',')
    stem_key = f'{image_type}_model_stem'
    channel_key = f'{image_type}_model_channels'
    output_dtype = cp.get(stage,'output_dtype')
    model_channels = cp.get('n2v', channel_key).split(',')   
    model_stem = cp.get('n2v', stem_key)
    do_min_subtraction = get_boolean( cp.get('n2v', 'do_min_subtraction') )
 
    logging.debug(f'image_type={image_type} channels={image_channels}')
    logging.debug(f'output_dtype={output_dtype} do_min_subtraction = {do_min_subtraction}')
    logging.debug(f'model basedir={basedir} model_stem={model_stem} model_channels={model_channels}')
    
    models = []
    for probe in model_channels:
        name = model_stem+probe
        logging.debug(f'loading model {name} from {basedir}')
        models.append( N2V(config=None, name=model_stem+probe, basedir=basedir) )
        
    logging.debug(f'got {len(models)} N2V models for {model_channels}')    
    logging.info(f'handling {len(infiles)} input files e.g. {infiles[0]}')

    for i, filename in enumerate( infiles ):
        (dirpath, base, label, ext) = split_path(os.path.abspath(filename))
        logging.debug(f'handling {filename}')
        #imgarray = imageio.imread(filename)
        imgarray = read_image(filename)        
        pred_image = []
        for j, img in enumerate(imgarray):
            try:
                logging.debug(f'{base}.{ext}[{i}] shape={img.shape} dtype={img.dtype}')
                pimg = models[j].predict(img, axes='YX')   # float32 prediction
                logging.debug(f'got model output: {base}.{ext}[{j}] shape={pimg.shape} dtype={pimg.dtype}')
            except Exception:
                logging.warning(f'ran out of models, appending channel [{j}] without prediction.')
                pimg = img
            # Match MATLAB n2vprocessing.py: (pred - pred.min()).astype(uint16) -- min-subtract on
            # the FLOAT prediction BEFORE the uint16 cast, for EVERY channel including appended
            # non-seq channels (DAPI/DIC). The previous order (cast to uint16 first, and no
            # min-subtraction on appended channels) left a +110 imaging / +72 DAPI brightness offset
            # vs MATLAB that propagated into hyb basecalling and the bardensr threshold calibration.
            if do_min_subtraction:
                pimg = pimg - pimg.min()
            pimg = pimg.astype(output_dtype)
            logging.debug(f'final dtype={pimg.dtype}')
            pred_image.append(pimg)
               
        logging.debug(f'done predicting {base}.{ext} {len(pred_image)} channels. ')
        newimage = np.dstack(pred_image)
        # produces e.g. shape = ( 3200,3200,5)
        newimage = np.rollaxis(newimage, -1)
        # produces e.g. shape = ( 5, 3200, 3200)        
        #tf.imwrite( outfile, newimage)
        outfile = outfiles[i]
        (outdir, file) = os.path.split(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            logging.debug(f'made outdir={outdir}')
        
        logging.info(f'writing to {outfile}')
        write_image( outfile , newimage )
        logging.debug(f'done writing {outfile} ')


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
      
    #outdir = os.path.abspath('./')
    #if args.outdir is not None:
    #    outdir = os.path.abspath(args.outdir)
    
    (outdir, file) = os.path.split(args.outfiles[0])
    logging.debug(f'ensuring outdir {outdir}')
    os.makedirs(outdir, exist_ok=True)
        
    datestr = dt.datetime.now().strftime("%Y%m%d%H%M")

    denoise_n2v( infiles=args.infiles, 
                 outfiles=args.outfiles,
                 stage=args.stage,  
                 cp=cp )
    
    logging.info(f'done processing output to {outdir}')

