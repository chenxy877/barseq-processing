#!/usr/bin/env python
#
# Calculate required bardensr per-experiment image processing thresholds/parameters. 
# Handle entire batched set of multiple tilesets. 
# 
import argparse
import itertools
import json
import logging
import math
import os
import pprint
import sys

import datetime as dt

from configparser import ConfigParser

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

import matplotlib.pylab as plt
import numpy as np

import bardensr
import bardensr.plotting

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

from joblib import Parallel, delayed


# Module-level per-FOV workers for joblib (loky) parallelism in calc_params_bardensr.
# They use the module-level `bardensr` and `bd_read_image_set` (re-imported per worker).
def _cp_fov_maxrc(tileset, R, C, cropf):
    return bd_read_image_set(tileset, R, C, cropf=cropf).max(axis=(1, 2, 3))

def _cp_fov_errmax(tileset, R, C, codeflat, median_max, pos0, trim, noisefloor_ini):
    img_norm = bd_read_image_set(tileset, R, C, trim=trim) / median_max[:, None, None, None]
    et = bardensr.spot_calling.estimate_density_singleshot(img_norm, codeflat, noisefloor_ini)
    return et[:, :, :, pos0].max(axis=(0, 1, 2))

def _cp_fov_fdr(tileset, R, C, codeflat, median_max, pos0, cropf, noisefloor_final, thresh_grid):
    img_norm = bd_read_image_set(tileset, R, C, cropf=cropf) / median_max[:, None, None, None]
    et = bardensr.spot_calling.estimate_density_singleshot(img_norm, codeflat, noisefloor_final)
    err_c_list, total_c_list = [], []
    for thresh1 in thresh_grid:
        spots = bardensr.spot_calling.find_peaks(et, thresh1, use_tqdm_notebook=False)
        err_c = sum((spots.j == k).to_numpy().sum() for k in pos0)
        err_c_list.append(int(err_c))
        total_c_list.append(int(len(spots) - err_c))
    return err_c_list, total_c_list


def calc_params_bardensr( infiles, outfiles, stage=None, cp=None):
    '''
    infiles and outfiles are *lists of lists* rather than usual 
    argparser parameter lists. e.g.  
    infiles = [ [ 't1c1','t1c2' ], ['t2c1', 't2c2' ] ]

    fdrthresh=0.05,
    trim=160,
    cropf=0.4,
    noisefloor_ini=0.01,
    
    noisefloor_final=0.05    
    '''
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'calc-params'

    # We know arity is globally single, so we can grab the outfile from the first set.  
    outfile = outfiles[0][0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')

    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))
    image_type = cp.get(stage, 'image_type')
    image_channels = cp.get(image_type, 'channels').split(',')
    logging.debug(f'resource_dir={resource_dir} image_type={image_type} image_channels={image_channels}')

    first_infile = infiles[0][0]

    logging.info(f'handling {len(infiles)} batches of input files e.g. {first_infile} ')
    (dirpath, base, label, ext) = split_path(os.path.abspath(first_infile))
    (prefix, subdir) = os.path.split(dirpath)
    logging.debug(f'dirpath={dirpath} base={base} ext={ext} prefix={prefix} subdir={subdir}')
    
    noisefloor_ini = cp.getfloat( stage, 'noisefloor_ini')
    noisefloor_final = cp.getfloat(stage, 'noisefloor_final')
    fdrthresh = cp.getfloat( stage, 'fdrthresh')
    trim = cp.getint(stage, 'trim')
    cropf = cp.getfloat(stage, 'cropf')
    logging.debug(f'noisefloor_ini={noisefloor_ini} trim={trim} cropf={cropf}')

    # load codebook TSV from resource_dir
    codebook_file = cp.get(stage, 'codebook_file')
    codebook_bases = get_config_list(cp, stage, 'codebook_bases')
    cfile = os.path.join(resource_dir, codebook_file)
    logging.info(f'loading codebook file: {cfile}')
    codebook = load_codebook_file(cfile)
    num_channels = len(codebook_bases) 
    logging.debug(f'loaded codebook TSV:\n{codebook} codebook_bases={codebook_bases}')    
    
    n_cycles = len(infiles[0])
    logging.info(f'Detected tilesets of {n_cycles} cycles.')
    (codeflat, R, C, J, genes, pos_unused_codes) = make_codebook_object(codebook, codebook_bases, n_cycles=n_cycles)
    logging.info(f'R={R} C={C} J={J} len(genes)={len(genes)} pos_unused_codes={pos_unused_codes}')
    # OUTPUT DICT
    param_outputs = {}

    # PARALLELISM: calc-params is ONE command over ALL (control) FOVs (it needs them
    # all to compute the global median_max + threshold), so the workflow's per-FOV
    # JobSet cannot split it. But the per-FOV et/find_peaks work IS independent, so we
    # parallelize the three passes with joblib over FOVs (loky processes; bardensr/TF
    # re-imported per worker via PYTHONPATH). n_cp_jobs from [bardensr] n_jobs.
    n_cp_jobs = min(len(infiles), cp.getint('bardensr', 'n_jobs', fallback=8))
    logging.info(f'calc-params parallelism: {n_cp_jobs} jobs over {len(infiles)} FOVs')
    pos0 = pos_unused_codes[0]

    # PASS 1: per-FOV max over each (cycle,channel) frame (center crop) -> global median_max.
    max_per_RC = Parallel(n_jobs=n_cp_jobs)(
        delayed(_cp_fov_maxrc)(tileset, R, C, cropf) for tileset in infiles)
    median_max = np.median(max_per_RC, axis=0)
    logging.info(f'median_max = {pprint.pformat(median_max, indent=4)}')

    # PASS 2: per-FOV base evidence over the unused codes -> intensity_thresh_ini.
    # (BUG FIX: index the code axis with pos_unused_codes[0] only -- the 2-tuple form
    # pulled gene 0/Calb1 into the base threshold, inflating it to 0.796 vs 0.676.)
    err_max = np.array(Parallel(n_jobs=n_cp_jobs)(
        delayed(_cp_fov_errmax)(tileset, R, C, codeflat, median_max, pos0, trim, noisefloor_ini)
        for tileset in infiles))
    thresh = np.median(np.median(err_max, axis=1))
    logging.info(f'intensity_thresh_ini={thresh}')

    # PASS 3: per-FOV FDR counts over the threshold grid. MATLAB bardensrbasecall.py
    # evaluates FDR over linspace(thresh-0.1, thresh+0.5, 10) but selects thresh_refined
    # from linspace(thresh-0.1, thresh+0.1, 10) at the crossing index (the wider search
    # lowers the crossing index -> lower threshold; matches MATLAB's 0.676).
    thresh_grid = np.linspace(thresh - 0.1, thresh + 0.5, 10)
    fdr_results = Parallel(n_jobs=n_cp_jobs)(
        delayed(_cp_fov_fdr)(tileset, R, C, codeflat, median_max, pos0, cropf, noisefloor_final, thresh_grid)
        for tileset in infiles)
    err_c_all = [e for (ec, tc) in fdr_results for e in ec]
    total_c_all = [t for (ec, tc) in fdr_results for t in tc]

    # CALCULATE FALSE DISCOVERY RATE, GIVEN N_SPOTS FOUND AT INTENSITY THRESHOLD
    err_c_all1 = np.reshape(err_c_all, [ len(infiles), 10 ])
    total_c_all1 = np.reshape(total_c_all, [ len(infiles), 10]) + 1
    fdr = err_c_all1 / len(pos_unused_codes[0]) * (len(genes)-len(pos_unused_codes[0])) / (total_c_all1)
    fdrmean = err_c_all1.mean(axis=0) / len(pos_unused_codes[0]) * (len(genes) - len(pos_unused_codes[0])) / (total_c_all1.mean(axis=0))
    thresh_refined=np.linspace( thresh-0.1, thresh+0.1, 10)[(fdrmean < fdrthresh).nonzero()[0][0]]

    #this is the new threshold optimized by targeted fdr value
    logging.info(f'intensity_thresh_refined = {thresh_refined}')
    
    thresh_refined = float(thresh_refined)
    noisefloor_ini = float(noisefloor_ini)
    noisefloor_final = float(noisefloor_final)

    param_outputs['intensity_thresh_refined'] = thresh_refined
    param_outputs['noisefloor_ini'] = noisefloor_ini
    param_outputs['noisefloor_final'] = noisefloor_final
    # DIAGNOSTIC dump (threshold-calibration investigation): base thresh, FDR curve, search grid.
    _thr = float(np.asarray(thresh).ravel()[0])
    _em = np.asarray(err_max, dtype=float)          # (nFOV, nUnused)
    param_outputs['_diag_intensity_thresh_ini'] = _thr
    param_outputs['_diag_fdrmean'] = np.asarray(fdrmean, dtype=float).ravel().tolist()
    param_outputs['_diag_thresh_search'] = np.linspace(_thr-0.1, _thr+0.5, 10).tolist()
    param_outputs['_diag_thresh_select'] = np.linspace(_thr-0.1, _thr+0.1, 10).tolist()
    param_outputs['_diag_selected_index'] = int(np.asarray(np.asarray(fdrmean).ravel() < fdrthresh).nonzero()[0][0])
    param_outputs['_diag_errmax_fov_median'] = np.median(_em.reshape(_em.shape[0], -1), axis=1).tolist()
    param_outputs['_diag_n_unused'] = int(_em.reshape(_em.shape[0], -1).shape[1])
    # Persist the GLOBAL per-frame (cycle,channel) normalizer computed across control
    # FOVs. MATLAB bardensrbasecall.py uses this same `maxmax` for BOTH threshold
    # calibration and the final per-FOV basecalling; basecall-geneseq must reuse it so
    # the calibrated threshold corresponds to the same density scale at basecall time.
    param_outputs['median_max'] = [float(x) for x in np.asarray(median_max).ravel()]
    logging.info(f"threshold {thresh_refined} with noise floor {noisefloor_final}")
    logging.info(f"param_outputs= {param_outputs} {len(infiles)} input tilesets. ")
    
    with open(outfile, 'w' ) as f:
        json.dump(param_outputs, f)
    logging.info(f'wrote params to {outfile}')
    return param_outputs

    
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
                    default='basecall-geneseq', 
                    type=str, 
                    help='stage we care calculating for. input learn')

    parser.add_argument('-i','--infiles',
                        metavar='infiles',
                        action='append',
                        nargs ="+",
                        type=str,
                        help='File[s] to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    action='append',
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

    
    param_outputs = calc_params_bardensr(   infiles=args.infiles, 
                                            outfiles=args.outfiles,
                                            stage=args.stage,  
                                            cp=cp   )
    print(param_outputs)
    
    logging.info(f'done processing output to {args.outfiles}')
 
 