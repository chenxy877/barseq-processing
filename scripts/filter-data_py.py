#!/usr/bin/env python
# Filter overlapping areas so we don't double-count anything...
# 
import argparse
import joblib
import logging
import math
import os
import re
import pprint
import sys
import datetime as dt
from configparser import ConfigParser

import anndata as ad
import numpy as np
from natsort import natsorted as nsort

gitpath=os.path.expanduser("~/git/barseq-processing")
sys.path.append(gitpath)

from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from barseq.core import *
from barseq.utils import *
from barseq.imageutils import *

def filter_data(infiles, outfiles, stage=None, cp=None):
    #    inputs: 'alldata.joblib'.  
    #    filt_neurons.joblib is main flag output. 
    if cp is None:
        cp = get_default_config()

    if stage is None:
        stage = 'filter-data'

    logging.info(f'infiles={infiles} outfiles={outfiles} stage={stage}')

    # We have heterogenous input files, so we need to confirm all are present, and 
    # figure out which is which. 
    # return order from select function will be alphabetical by key name.  
    input_map = {   'alldata'       :  'alldata.joblib',
                    'processeddata' :  'processeddata.joblib',
                }
    (alldata_file, processeddata_file) = select_input_files(infiles, input_map)

    # We know arity is single, so we can grab the outfile
    # primary outfile is filt_neurons.joblib
    #  
    outfile = outfiles[0]
    (outdir, file) = os.path.split(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logging.debug(f'made outdir={outdir}')        
    logging.info('infile={infile} outfile={outfile}')

    # Get parameters
    logging.info(f'handling stage={stage} to outdir={outdir}')
    resource_dir = os.path.abspath(os.path.expanduser( cp.get('barseq','resource_dir')))

    project_id = cp.get( 'project','project_id')
    rescale_factor=cp.getfloat(stage, 'rescale_factor')
    px = cp.getfloat( stage, 'px')
    box_half_width_um = cp.getfloat( stage, 'box_half_width_um')
    search_radius_um = cp.getint( stage, 'search_radius_um')

    logging.info(f'stage={stage} rescale_factor={rescale_factor} px={px} box_half_width_um={box_half_width_um} search_radius_um = {search_radius_um} ')

    data = joblib.load(alldata_file)
    pr = px / rescale_factor
    overlap_half_width = np.round(box_half_width_um / pr)
    search_radius = np.round(search_radius_um / pr)
    neurons = data['neurons']
    center_x=neurons['pos10x_x']
    center_y=neurons['pos10x_y']
    exp_mat=neurons['expmat'].todense() # I should do it in csr rather than dense--memory efficient

    xmin = center_x - overlap_half_width
    xmax = center_x + overlap_half_width
    ymin = center_y - overlap_half_width
    ymax = center_y + overlap_half_width
    c = np.column_stack((center_x,center_y))
    num_slices = np.unique(neurons['slice'])
    filt_neurons = {}
    id_to_keep_all =[]
    
    for uslice in num_slices:       
        idx_slice = neurons['slice'] == uslice
        ## this will crash for big population of cells
        # dist=cdist(c[idx_slice],c[idx_slice],'euclidean')
        # dist_nearest=(dist<search_radius)
        # [cell_id,nearest_neigh_id]=np.nonzero(dist_nearest)
        # fov_neigh=neurons['fov'][idx_slice][nearest_neigh_id]
        # fov_cell=neurons['fov'][idx_slice][cell_id]
        # sel_cells_id=fov_cell!=fov_neigh
        # distances_neigh=dist[cell_id[sel_cells_id],nearest_neigh_id[sel_cells_id]]

        tree = cKDTree( c[idx_slice] )
        sparse_dist = tree.sparse_distance_matrix(tree, 
                                                  max_distance=search_radius, 
                                                  output_type='coo_matrix')
        cell_id = sparse_dist.row
        nearest_neigh_id = sparse_dist.col
        dist_vals = sparse_dist.data
        fov_neigh = neurons['fov'][idx_slice][nearest_neigh_id]
        fov_cell = neurons['fov'][idx_slice][cell_id]
        sel_cells_id = fov_cell != fov_neigh
        distances_neigh = dist_vals[sel_cells_id] 
    
        search_cells_id = cell_id[sel_cells_id]
        search_neighbors_id = nearest_neigh_id[sel_cells_id]
        search_dist = distances_neigh
        
        id_overlap=((((xmin[idx_slice][search_cells_id]<xmin[idx_slice][search_neighbors_id])&(xmin[idx_slice][search_neighbors_id]<xmax[idx_slice][search_cells_id])) |
                     ((xmin[idx_slice][search_cells_id]<xmax[idx_slice][search_neighbors_id])&(xmax[idx_slice][search_neighbors_id]<xmax[idx_slice][search_cells_id]))) & 
                     (((ymin[idx_slice][search_cells_id]<ymin[idx_slice][search_neighbors_id])&(ymin[idx_slice][search_neighbors_id]<ymax[idx_slice][search_cells_id])) | 
                     ((ymin[idx_slice][search_cells_id]<ymax[idx_slice][search_neighbors_id])&(ymax[idx_slice][search_neighbors_id]<ymax[idx_slice][search_cells_id]))))
    
        #id_overlap = search_dist < overlap_half_width
        #     
        overlap_cells_id=search_cells_id[id_overlap]
        overlap_neighbors_id=search_neighbors_id[id_overlap]
        overlap_distance=search_dist[id_overlap]
        # for i,idc in enumerate(overlap_cells_id):
        #     print(f"Cell {neurons['id'][idc]} in fov {neurons['fov_names'][neurons['fov'][idc]]} is matched to cell {neurons['id'][overlap_neighbors_id[i]]} in fov {neurons['fov_names'][neurons['fov'][overlap_neighbors_id[i]]]} with distance {overlap_distance[i]}")
        
        logging.info(f"Total cells: {len(center_x[idx_slice])}")
        logging.info(f"Cells in overlap pairs: {len(np.unique(overlap_cells_id))}")
        logging.info(f"Fraction of cells in overlaps: {len(np.unique(overlap_cells_id))/len(center_x[idx_slice]):.2%}")       

        n_cells_slice = int(idx_slice.sum())
        adj = csr_matrix((np.ones(len(overlap_cells_id)), 
                         (overlap_cells_id, overlap_neighbors_id)), 
                         shape=(n_cells_slice, n_cells_slice))
        adj = adj + adj.T  # symmetrize
        n_components, comp_labels = connected_components(adj, directed=False)
        
        total_exp_cell = np.asarray(np.sum(exp_mat[idx_slice,:], axis=1)).flatten()
        
        is_removed = np.zeros(n_cells_slice)
        for comp in range(n_components):
            members = np.where(comp_labels == comp)[0]
            if len(members) <= 1:
                continue
            best = members[np.argmax(total_exp_cell[members])]
            is_removed[members] = 1
            is_removed[best] = 0
        
        id_to_keep = is_removed == 0
        id_to_keep_all.append(id_to_keep)
    id_to_keep_all=np.concatenate(id_to_keep_all)    
    expmat=coo_matrix(exp_mat[id_to_keep_all,:])
    filt_neurons['expmat']=expmat
    filt_neurons['id']=neurons['id'][id_to_keep_all]
    filt_neurons['pos10x_x']=neurons['pos10x_x'][id_to_keep_all]
    filt_neurons['pos10x_y']=neurons['pos10x_y'][id_to_keep_all]
    filt_neurons['pos40x_x']=neurons['pos40x_x'][id_to_keep_all]
    filt_neurons['pos40x_y']=neurons['pos40x_y'][id_to_keep_all]
    filt_neurons['slice']=neurons['slice'][id_to_keep_all]
    filt_neurons['genes']=neurons['genes']
    filt_neurons['fov']=neurons['fov'][id_to_keep_all]
    filt_neurons['fov_names']=neurons['fov_names']

    output_data = {"filt_neurons":filt_neurons, "removecells_all":id_to_keep_all}

    logging.info(f"search_radius: {search_radius}")
    logging.info(f"cross-FOV pairs: {sel_cells_id.sum()}")
    logging.info(f"overlap pairs: {id_overlap.sum()}")
    logging.info(f"cells removed: {int((id_to_keep_all==0).sum())}")   
    logging.info(f'OVERLAPPING CELLS REMOVED, {np.sum(id_to_keep_all)} {100 * np.sum(id_to_keep_all)/len(center_x)} % cells kept out of {len(center_x)}--PROCESSING FINISHED')

    dfexpmat = pd.DataFrame( expmat.todense() )   
    of = os.path.join( outdir, f'{project_id}.filt_cellsbygenes.tsv')
    dfexpmat.to_csv(of, sep='\t') 
    logging.info(f'Wrote cells X genes matrix to {of}')

    logging.info(f'Writing overall output to {outfile}')
    joblib.dump(output_data, outfile)

    aof = os.path.join( outdir, f'{project_id}.filt_neuron.anndata.h5ad')
    logging.info(f'Also attemption AnnData output to {aof} ')
    try: 
        X = filt_neurons['expmat']
        if sparse.issparse(X):
            X = X.tocsr()

        # --- obs dataframe ---
        obs = pd.DataFrame( 
            {   'cell_id': np.asarray( filt_neurons['id']).ravel(), 
                'pos_x': np.asarray(filt_neurons['pos10x_x']).ravel(),
                'pos_y': np.asarray(filt_neurons['pos10x_y']).ravel(),
                'slice': np.asarray(filt_neurons['slice']).ravel(),         
            } )    

        # --- var dataframe ---
        #genes=[ g[0][0].item() for g in filt_neurons['genes']]
        genes = list( filt_neurons['genes']['gene'] )
        genes = np.array(genes)

        var = pd.DataFrame(index=genes)
        var["gene_symbol"] = genes

        # --- make AnnData ---
        adata = ad.AnnData(X=X, obs=obs, var=var)

        # optional: put spatial coordinates in obsm like Scanpy expects
        adata.obsm["spatial"] = obs[["pos_x", "pos_y"]].to_numpy()

        aof = os.path.join( outdir, f'{project_id}.filt_neuron.anndata.h5ad')
        logging.info(f'Writing anndata to {aof}')
        adata.write_h5ad(aof)

    except Exception as e:
        logging.error(f"Problem outputting AnnData: '{e}'")

    logging.info(f'Done.')


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
                        required=True,
                        nargs ="+",
                        type=str,
                        help='All image files to be handled.') 

    parser.add_argument('-o','--outfiles', 
                    metavar='outfiles',
                    required=True, 
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

    filter_data( infiles=args.infiles, 
                            outfiles=args.outfiles,
                            stage=args.stage,  
                            cp=cp )
    logging.info(f'done processing output to {args.outfiles[0]}')
