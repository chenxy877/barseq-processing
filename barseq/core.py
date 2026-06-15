import itertools
import logging
import math
import os
import pprint
import re
import sys
import time
import traceback

from collections import defaultdict
from configparser import ConfigParser

from natsort import natsorted as nsort

import scipy
import numpy as np
import pandas as pd
import tifffile as tif

from scipy.sparse import dok_matrix, csr_matrix, lil_matrix, csc_matrix, coo_matrix, bsr_matrix

from barseq.utils import *

class BarseqExperiment():
    '''
        Methods and data structure to keep and deliver sets of files in groupings as needed.
        Generates maps of stage-to-stage file relationships as required.  
        Centralized metadata. 
        Abstract out mode identifiers from directory names, paths from cycles, and names from position tilesets. 
        
        ddict   (directory dict)    keys = modes, values= list of cycle directories. 
        tdict   (tile dict)         keys = modes, values= list of lists:  cycles, tiles   
        pdict   (position dict)      
        
        All real file paths are stored as relative paths in the object, to allow mapping from 
        subdir to subdir by steps in the pipeline. 
        
        Goal is to 
        1. Fully validate (for missing input files) before proceeding. 
        2. Allow retrieving...
            -- All tiles in flat form, grouped by chunksize (for flat processing). 
            -- All tiles, grouped by position, and grouped by chunksize (for stitching).
            -- All tiles within a mode, but across cycles, for a tilename (for registration) 
        3. Allow checking for existing output? 

        NOMENCLATURE
        
        mode     top-level category of image subsets
        cycle    a mode has one or more cycles, consisting of positions made up of images
        image    a single image file
        tile     a set of images that represent a single FOV across cycles
        position a set of adjacent images across cycles 

        Assumes path/naming hierarchy:
        
        EXP123/

            M1C1/           M1C2/               M2C1/
                P1T1.<ext>     P1T1.<ext>           P1T1.<ext>
                P1T2.<ext>     P1T2.<ext>           P1T2.<ext>
                P2T1.<ext>     P2T1.<ext>           P2T1.<ext>
                P2T2.<ext>     P2T2.<ext>           P2T2.<ext>

            M1/                                 M2/
                P1.<label>.<ext>                    P1.<label>.<ext>
                P2.<label>.<ext>                    P2.<label>.<ext>
 
        Key methods/ conventions. 
        return elements are relative PATHS under EXP123, not simply file names. 
        This allows retention of subdirectory hierarchy on outputs. 
         
        get_Xlist  -> flat list
        get_Xset   -> list of lists, structured appropriately. 
        
        get_filelist()
            Flat list of all files. 
            Unit of work: individual file
            [ M1C1/P1T1, M1C1/P1T2, M1C1/P2T1, M1C1/P2T2, 
              M1C2/P1T1, M1C2/P1T2, M1C2/P2T1, M1C2/P2T2,
              M2C1/P1T1, M2C1/P1T2, M2C1/P2T1, M2C1/P2T2
             ]
        
        get_cycleset( mode=M1)  
            Files grouped by cycle, otherwise unstructured. 
            Unit of work: individual file. 
        
            [ [ M1C1/P1T1, M1C1/P1T2, M1C1/P2T1, M1C1/P2T2 ], 
              [ M1C2/P1T1, M1C2/P1T2, M1C2/P2T1, M1C2/P2T2 ] 
            ]        
        
        get_positionset( mode=M1 )
            Files grouped by position value, otherwise unstructured. 
            Unit of work: all files for position (e.g. for stitching). 
            [ [ M1C1/P1T1, M1C1/P1T2], [ M1C1/P2T1, M1C1/P2T2 ], 
              [ M1C2/P1T1, M1C2/P1T2], [ M1C2/P2T1, M1C2/P2T2 ], 
            ]
        
        get_tileset( mode=M1 )
            Files grouped by tile value, ordered by cycle.
            Unit of work: all images for given tile, typically within mode. 
                e.g. for registration. 
            [  [ M1C1/P1T1, M1C2/P1T1 ], 
               [ M1C1/P1T2, M1C2/P1T2 ],
               [ M1C1/P2T1, M1C2/P2T1 ],
               [ M1C1/P2T2, M1C2/P2T2 ]
            ] 
    
        CONVENTIONS
    

    '''
    
    def __init__(self, indir, outdir, cp=None):
        '''
        @arg indir     overall input data directory
        @arg outdir    ovarall output working directory
        @arg cp        experiment config
                
        '''       
        self.cp = cp
        if cp is None:
            self.cp = get_default_config()
        self.inputdir = os.path.abspath( os.path.expanduser(indir))
        self.outputdir = os.path.abspath( os.path.expanduser(outdir)) 
        self.modes = [ x.strip() for x in self.cp.get('experiment','modes').split(',') ]

        # cache parsed file map trees. 
        # input key is 'input'
        self.stageinfo = {}
        
        (ddict, cdict, pdict) = self.parse_stage_dir()
        #logging.debug(f'ddict = {ddict} cdict={cdict} pdict={pdict}')
        logging.debug(f'ddict len={len(ddict)} cdict len={len(cdict)} pdict len={len(pdict)}')
        
        self.ddict = ddict
        self.cdict = cdict
        self.pdict = pdict

        logging.debug('BarseqExperiment metadata object intialized. ')

    def __repr__(self):
        s = f'BarseqExperiment: \n'
        modes = list( self.modes )
        modes.sort()
        for mode in modes :
            ncyc = len(self.cdict[mode])
            ntiles = 0
            for cyc in self.pdict[mode]:
                for p in list( cyc.keys()):
                    ntiles += len(cyc[p].flatten())          
            s += f'  mode={mode}\tncycles={ncyc}\tntiles={ntiles}\n'
            cyclelist = self.pdict[mode]
            for i, cycle in enumerate( cyclelist):
                s += f'    cycle[{i}]\n'
                skeys = list( cycle.keys())
                skeys.sort()
                for p in skeys:
                    (x,y) = cycle[p].shape
                    s += f'     pos={p} tiles={x*y} [{x}x{y}]\n'
        for stage in self.stageinfo.keys():
            s += f'\n'
            s += f'Stage Information:\n'
            s += f'   [{stage}]\n'
            (ddict, cdict, pdict) = self.stageinfo[stage]
            modes = list( cdict.keys())
            modes.sort()
            s += f'    modes={modes}\n'
            for mode in modes:
                if ( len( pdict[mode]) > 0 ) or (len( cdict[mode])  > 0 ) :
                    s += f'     [{mode}]\n'
                    s += f'         n_positions = {len( pdict[mode][0])}\n'
                    s += f'         n_cycles = {len(cdict[mode])}\n'
        return s


    def parse_stage_dir(self, stage=None):
        '''
        Top-level combined method, handling dirs, cycles, and positions. 
        '''
        re_list = []
        pdict = {}
        ddict = {}

        modes = get_config_list(self.cp, 'experiment', 'modes')

        for mode in modes:
            p = re.compile( self.cp.get( 'barseq',f'{mode}_regex'))
            re_list.append(p)
            pdict[p] = mode 
            ddict[mode] = []
        
        if stage is None:
            parse_dir = self.inputdir
            file_regex = get_config_list( self.cp, 'barseq' , 'file_regex')
        else:
            stagedir = self.cp.get(stage, 'stagedir' )
            parse_dir = os.path.join( self.outputdir, stagedir )
            file_regex = get_config_list( self.cp, stage , 'file_regex')
        logging.debug(f'parse directory is {parse_dir}')
        
        dlist = os.listdir(parse_dir)
        dlist.sort()
        for d in dlist:
            for p in pdict.keys():
                if p.search(d) is not None:
                    k = pdict[p]
                    ddict[k].append(d)
        logging.debug(f'directory dict = {ddict}')
      
        cdict = {}
        for mode in self.modes:
            cdict[mode] = []  # list of lists
            for d in ddict[mode]:
                cyclelist = []
                cycledir = f'{parse_dir}/{d}'
                logging.debug(f'listing cycle dir {cycledir}')
                flist = os.listdir(cycledir)
                flist.sort()
                fnlist = []
                n_passed = 0
                base_passed = ''
                n_failed = 0
                base_failed = ''
                for f in flist:
                    # we don't use split_path because we want to match against whole file basename
                    # including label (if any)
                    dp, filename = os.path.split(f)
                    base, ext = os.path.splitext(filename)
                    ext = ext[1:]
                    #m = re.search(file_regex, base)
                    m = search_regex_list_any(file_regex , base )
                    if m is not None:
                        n_passed += 1
                        base_passed = base
                        rfile = f'{d}/{base}.{ext}'
                        cyclelist.append(rfile)
                    else:
                        n_failed += 1
                        base_failed = base
                logging.debug(f'dir scan: file_regex={file_regex}  n_passed={n_passed} n_failed={n_failed} E.g. {base_failed}')
                cdict[mode].append(cyclelist)        

        pdict = {}
        for mode in self.modes:
            pdict[mode] = []
            cycfilelist = cdict[mode]
            for i, cycle in enumerate( cycfilelist ):
                #logging.debug(f'creating cycle dict for {mode}[{i}]')
                cycdict = {}
                n_passed = 0
                base_passed = ''
                n_failed = 0
                base_failed = ''
                passed_index = 0
                for rfile in cycle:
                    posarray = None
                    afile = os.path.abspath(f'{parse_dir}/{rfile}')
                    # logging.debug(f'handling a file: {afile}')
                    # we don't use split_path because we want the whole file basename
                    # including label (if any)
                    dp, filename = os.path.split(afile)
                    base, ext = os.path.splitext(filename)
                    ext = ext[1:]
                    #m = re.search(file_regex, base)
                    m = search_regex_list_any(file_regex , base )
                    if m is not None:
                        n_passed += 1
                        base_passed = base
                        try:
                            pos = m.group(1)
                            x = m.group(2)
                            y = m.group(3)
                            x = int(x)
                            y = int(y)
                            #logging.debug(f'mode={mode} cycle={i} pos={pos} x={x} y={y} type(pos)={type(pos)}')
                            #logging.debug(f'cycdict.keys() = {list( cycdict.keys() )}')
                            pos = str(pos).strip()
                            try:    
                                posarray = cycdict[pos] 
                                #logging.debug(f'success. got posarray for cycle[{i}] position {pos}')
                            
                            except KeyError:
                                #logging.debug(f'KeyError: creating new position dict for {pos} type(pos)={type(pos)}')
                                cycdict[pos] = SimpleMatrix()
                                #logging.debug(f'type = {type( cycdict[pos]) }')
                                
                            fname = f'{rfile}'
                            #logging.debug(f"saving posarray[{x},{y}] = '{rfile}'")                            
                            cycdict[pos][x,y] = fname

                        except IndexError:
                            # Matching name may not have Position information. 
                            # Put them all into a single 1-D position. 
                            logging.warning(f'no matching groups for {base}')
                            fname = f'{rfile}'
                            try:
                                cycdict[0][passed_index,0] = fname
                                logging.debug(f'Setting value in existing matrix. file={fname}')
                            except KeyError:
                                logging.debug(f'creating new SimpleMatrix for file={fname}')
                                cycdict[0] = SimpleMatrix()
                                cycdict[0][passed_index,0] = fname
                            passed_index += 1 
                    else:
                        n_failed += 1
                        base_failed = base
                logging.debug(f'dir scan: file_regex={file_regex}  n_passed={n_passed} n_failed={n_failed} E.g. {base_failed}')    
                pdict[mode].append(cycdict)
                
            #logging.debug(f'fixing sparse matrices...')
            for i, cycdict in enumerate( pdict[mode]):
                pkeys = list(cycdict.keys())
                pkeys.sort()
                for p in pkeys:
                    sm = cycdict[p]
                    #logging.debug(f"fixing sarray {mode} cycle[{i}] position '{p}' type={type(sm)} ")
                    pnew = sm.to_ndarray()
                    #logging.debug(f"pnew type={type(pnew)} ")
                    cycdict[p] = pnew
        
        if stage is not None:
            logging.debug(f'caching stage info for {stage}')
            self.stageinfo[stage] = (ddict, cdict, pdict)
        return (ddict, cdict, pdict)    


    def get_stage_map(self, 
                            mode='geneseq', 
                            stage=None, 
                            label=None, 
                            ext=None, 
                            arity='single',
                            instage=None,
                            instage_mode=None,
                            strip_base=False,
                            maptype='cycle'
                            ) :
        '''
        Abstracted input-output file map creation. To replace per-maptype code.  

        @arg mode       Get map for modality, None means all. 
        @arg stage      Output stage name. 
        @arg label      Output extra label (before extension) dot-separated. 
        @arg ext        Output file(s) extension. 
        @arg arity      Arity from input to output. parallel=one-to-one single=many-to-one
        @arg instage    Input stage name. None= initial input.  
        @arg strip_base Remove base filename from output.  True|False      
        @arg maptype    Kind of map to generate.  cycle|position|tileset|filelist  
        
        ''' 
        logging.info(f'mode={mode} stage={stage} label={label} ext={ext} arity={arity} instage={instage} strip_base={strip_base}')
                       
        if mode is None:
            mode_list = list(self.cdict.keys())
            mode_list.sort()
            mode = mode_list

        stagedir = self.cp.get(stage, 'stagedir')
        # instage_mode = get_config_list(self.cp, stage , 'instage_modes')
       
        logging.info(f"get_stage_files(mode={instage_mode}, stage='{instage}', maptype='{maptype}')")
        infile_set = self.get_stage_files(mode=instage_mode, stage=instage, maptype=maptype)

        # Holds list of input-output sets, to be used one per process...
        output_list = []
        output_elem = None

        if arity == 'parallel':
            # for parallel, each input gets output. 
            for infile_list in infile_set:
                input_list = []
                outfile_list = []
                for rpath in infile_list:
                    input_list.append(rpath)
                    if (ext is not None) or (label is not None):
                        (subdir, base, current_label, current_ext) =  parse_rpath(rpath)
                        if ext is None:
                            ext = current_ext
                        if label is not None:
                            out_rpath = os.path.join( subdir, f'{base}.{label}.{ext}')
                        else:
                            out_rpath = os.path.join( subdir, f'{base}.{ext}')
                        outfile_list.append( out_rpath )
                    else:
                        outfile_list.append( rpath )
                output_list.append( ( input_list, outfile_list) )        
                
        elif arity == 'single':
            # Use first input rpath as model for output_rpath
            # Assume mode output dir (not numbered cycle dir)
            # Assume stage mode as correct mode directory
            # Flatten inputs to single input list. 
            for infile_list in infile_set:
                input_list = []
                for rpath in infile_list:
                    input_list.append(rpath)

                mode = instage_mode[0]
                (subdir, base, current_label, current_ext) = parse_rpath( infile_list[0])
                if (ext is not None) or (label is not None):
                    if ext is None:
                        ext = current_ext
                    if label is not None:
                        if strip_base:
                            # stripping base only makes sense if there is a label.
                            # and if arity=single
                            output_elem = os.path.join( mode, f'{label}.{ext}')
                        else:
                            output_elem = os.path.join( mode, f'{base}.{label}.{ext}')
                    else:
                        output_elem = os.path.join( mode, f'{base}.{ext}')
                else:
                    output_elem = os.path.join( mode, f'{base}.{ext}')
                        
                logging.debug(f'filelist output={( infile_list, output_elem)}')        
                output_list.append( ( infile_list , [ output_elem ] )  )
        logging.debug(f'made list of {len(output_list)} filemaps')     
        return output_list    



    def get_stage_files(self, mode=None, stage=None, maptype='filelist'):
        '''
        Abstracted file set creation. To replace per-maptype code.  

        @arg mode       Get map for modality. Can be singleton or list. None means all. 
        @arg stage      Output stage name.
        @arg maptype    Kind of file set to get.  cycle|position|tileset|filelist

        Returns list of lists (L1 of L2) 
        cycle       -> L1 elements are cycles, L2 elements are files within cycle. 
        position    -> L1 elements are positions, L2 elements are files within position.
        tileset     -> L1 elements are tiles, L2 elements are files across cycles.
        
        Returns flat list
        filelist    -> Elements are all files in experiment of mode.

        I mode is a list, output is still list of lists (no hierarchy)
               
        '''
        logging.info(f'mode={mode} stage={stage} maptype={maptype}')
        outlist = []
     
        # Set stage and modes
        # Ensure it is a list
        if mode is None:
            modes = self.modes
        elif type(mode) == str:
            modes = [mode]
        else:
            modes = mode

        if stage is None:
            # Use input dataset
            cdict = self.cdict
            pdict = self.pdict
            ddict = self.ddict
        else:
            # Parse stage dir.
            (ddict, cdict, pdict) = self.parse_stage_dir( stage )

        if maptype == 'cycle':
            '''
            single mode: simple case for single mode. one file set for each cycle.  
            
            multiple modes: merge cycles by ordered set. 
            i.e. a cycle set may include files from multiple modes, first with first, 
            second with second, etc. 
            '''
            logging.debug(f'getting cycle files')
            if len(modes) == 1:
                m = modes[0]
                logging.debug(f'cycle: handling mode={m}')
                for cycle_filelist in cdict[m]:
                    outlist.append(cycle_filelist)            
    
            elif len(modes) > 1:
                max_len = 1
                for m in modes:
                    mlen = len(cdict[m])
                    if mlen > max_len:
                        max_len = mlen
                idx = 0
                while idx < max_len:
                    cycle_filelist = []
                    for m in modes:
                        try:
                            for rpath in cdict[m][idx]:
                                cycle_filelist.append(rpath)
                        except IndexError:
                            pass
                    outlist.append(cycle_filelist)
                    idx += 1
                    
        elif maptype == 'position':
            logging.debug(f'getting position files')
            for m in modes:
                logging.debug(f'positionset: handling mode={m}')
                mode_cyc_list = pdict[m]
                logging.debug(f'positionset: handling mode_cyc_list len={len(mode_cyc_list)} type={type(mode_cyc_list)}') 
                for position_dict in mode_cyc_list:
                    logging.debug(f'positionset: handling position_dict=\n{position_dict}')
                    for pos_key in list( position_dict.keys() ):
                        logging.debug(f'handling pos_key={pos_key}')
                        tlist = []
                        for t in position_dict[pos_key].flatten():
                            #t = t.decode('UTF-8')                       
                            t = str(t)
                            tlist.append(t)
                        outlist.append(tlist)

        elif maptype == 'tileset':
            logging.debug(f'getting tile files')
            for file_index in range(0, len(cdict[modes[0]][0])):
                file_list = []
                for m in modes:    
                    for cyc in cdict[m]:
                        file_list.append(cyc[file_index])
                    outlist.append(file_list)

        elif maptype == 'filelist':
            logging.debug(f'getting all files in flat list.')
            for m in modes:
                for cyc in pdict[m]:
                    for p in list( cyc.keys()):
                        for t in cyc[p].flatten():
                            try:
                                t = t.decode('UTF-8')
                            except:
                                t = str(t)
                            outlist.append(t)
            # Make output list of lists to be compatible with other maptypes
            # when processed via process_stage_map()
            outlist = [ outlist ]
        else:
            logging.error(f'Invalid maptype: {maptype}')
        logging.debug(f'got output list len={len(outlist)}')
        return outlist 


    def _fix_sparse(self, sarray):
        '''
        remove empty rows and columns. convert to normal ndarray. 
        '''
        logging.debug(f'input type = {type(sarray)}')
        darray = sarray.toarray()
        nan_cols = np.all(darray == b'', axis = 0)
        nan_rows = np.all(darray == b'', axis = 1)        
        darray = darray[:,~nan_cols]
        darray = darray[~nan_rows,:]
        return darray
          
    def validate(self):
        '''
            -- confirms that there images corresponding to all tiles in all cycles of each mode. 
            --             
            @return True if valid, False otherwise   logs warnings as check is made. 
        '''
        return True


def search_regex_list_any( regex_list, test_string  ):
    '''
    apply all regexes in regex_list to test_string 
    return first successful match
    otherwise return None
    '''
    re_match = None
    for regex in regex_list:
        m = re.search(regex, test_string)
        if m is not None:
            return m
    return re_match


def get_default_config():
    dc = os.path.expanduser('~/git/barseq-processing/etc/barseq.conf')
    cp = ConfigParser()
    cp.read(dc)
    return cp
    
def get_script_dir():
    logging.debug(f'getting current script name {sys.argv[0]}')
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    logging.debug(f'script_dir = {script_dir}')
    return script_dir

def process_stage_map(indir, outdir, bse, stage=None, cp=None, force=False):
    '''
    Abstracted generic function. Retrieves maptype from config. 
    
    @arg indir          Top-level input directory (with cycle dirs below)
    @arg outdir         Outdir is top-level out directory (with cycle dirs below) UNLIKE stage_all_images
    @arg bse            bse is BarseqExperiment metadata object with relative file/mode layout
    @arg stage          Pipeline stage label in cp.
    @arg cp             ConfigParser object to refer to.    
 
    @return None

    Retrieves correct input file(s) to output file(s) mape. 
    Optionally allows one template to process input against.
    Checks for output existence. 
    Generates commmand line(s). 
    Runs commands in proper environment in desired thread count. 
     
    '''
    if cp is None:
        cp = get_default_config()

    logging.info(f'handling stage={stage} indir={indir}, outdir={outdir} force={force}')

    # file mapping parameters
    arity = cp.get(stage, 'arity')
    instage = get_config_none(cp,  stage, 'instage')
    instage_dir = None
    if instage is not None:
        instage_dir = cp.get(instage, 'stagedir')
    instage_mode = get_config_list(cp, stage, 'instage_modes')
    label = get_config_none(cp, stage, 'label')
    ext = get_config_none(cp, stage, 'ext')
    maptype = cp.get(stage, 'maptype')
    mode = get_config_list(cp, stage, 'modes')
    strip_base = cp.getboolean( stage, 'strip_base')

    # Tool execution information.
    tool = cp.get( stage ,'tool')
    n_jobs = int( cp.get( tool, 'n_jobs') )
    n_threads = int( cp.get(tool, 'n_threads') )
    logging.debug(f'handling stage={stage} indir={indir} outdir={outdir} ')
    
    # Get grouped file mapping(s)
    logging.info(f"get_stage_map( mode={mode}, stage='{stage}', maptype='{maptype}', label='{label}', ext='{ext}', arity='{arity}', instage='{instage}', instage_mode={instage_mode}, strip_base={strip_base})")
    file_map = bse.get_stage_map( mode=mode, 
                                       stage=stage, 
                                       label=label,
                                       ext=ext,
                                       arity=arity, 
                                       instage=instage,
                                       instage_mode=instage_mode,
                                       strip_base = strip_base,
                                       maptype=maptype
                                    ) 
    logging.debug(f'file_map= {file_map}')

    # Make command line(s)    
    command_list = []
    n_cmds = 0
    command_list = make_command_list( file_map, stage=stage, bse=bse, indir=indir, outdir=outdir, cp=cp)
    n_cmds = len(command_list)
    logging.info(f'created {n_cmds} commands for mode={mode}')
    
    if n_cmds > 0:
        run_jobs_local(command_list, n_jobs)
    logging.info(f'done with stage={stage}...')


def run_jobs_local(command_list, n_jobs):
        '''
        Run commands in sub-processes on local computer.
        '''
        n_cmds = len(command_list)
        logging.info(f'Creating jobset for {n_cmds} jobs on {n_jobs} CPUs ')    
        jstack = JobStack()
        jstack.setlist(command_list)
        jset = JobSet( max_processes = n_jobs, jobstack = jstack)
        logging.info(f'Running jobs...')
        jset.runjobs()
        if not jset.all_jobs_succeeded():
            logging.error('Job failure. ')
            raise NonZeroReturnException(f'Some job failed.')
        else:
            logging.info(f'All jobs succeeded.')

def make_command_list(file_map, stage, bse, indir, outdir, cp):
    '''

    CHUNKED version. 

    make command list from file_map
    create config
    check for output existence, skipping if all present. 
    
    '''
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)

    # Batch multiple filemaps into single command
    # Using multiple --infiles --outfiles and --template args if needed.  
    #
    n_chunks = cp.getint(stage, 'n_chunks', fallback=1)

    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    current_env = os.environ['CONDA_DEFAULT_ENV']
    instage = get_config_none(cp, stage, 'instage')
    instage_dir = None
    if instage is not None:
        instage_dir = cp.get(instage, 'stagedir')

    log_arg = ''
    log_level = logging.getLogger().getEffectiveLevel()
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    mode = get_config_list(cp, stage, 'modes' )
    num_cycles = int(cp.get(stage, 'num_cycles'))
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    script_base = cp.get(stage, 'script_base')
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    stagedir = cp.get(stage, 'stagedir')

    strip_base = cp.getboolean(stage, 'strip_base')
    template_mode = get_config_none(cp, stage, 'template_mode')
    template_source = get_config_none(cp, stage, 'template_source')
    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    chunked_filemaps = [ file_map[i : i + n_chunks] for i in range(0, len(file_map), n_chunks) ]

    # Define template file(s), if requested.
    template_file_list = None
    template_stagedir = None 
    if template_mode is not None:
        template_stagedir = cp.get(template_source, 'stagedir')
        # Only tileset currently makes sense for templates. 
        template_fileset_list = bse.get_stage_files( template_mode, stage=template_source, maptype='tileset' )
        logging.debug(f'template_stagedir={template_stagedir} template_fileset_list = {template_fileset_list}')
        chunked_templates = [ template_fileset_list[i :  i + n_chunks] for i in range(0, len(template_fileset_list), n_chunks) ]

    # Create command line(s) for mapping sets
    command_list = []

    for i, fmap_batch in enumerate( chunked_filemaps):
        logging.debug(f'handling file group {i}')
        cmd = []
        if conda_env != current_env:
            logging.debug(f'different envs. user conda run...')
            cmd = ['conda','run',
                        '-n', conda_env , 
                        'python', script_path,
                        log_arg, 
                        '--config' , runconfig ,                            
                        ]
        else:
            logging.debug(f'same envs needed, run direct...')
            cmd = ['python', script_path,
                        log_arg,
                        '--config' , runconfig, 
                        ]    
        cmd.append('--stage')
        cmd.append(f'{stage}')

        # 
        for j, fmap in enumerate( fmap_batch ):
            (input_list, output_list) = fmap
            logging.debug(f'stage = {stage} file_index={j} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
            logging.debug(f'input = {input_list} output = {output_list}')

            if template_mode is not None:            
                cmd.append( f'--template ')
                if template_source == 'input':
                    #template_file = os.path.join(indir, template_stagedir, template_fileset_list[i][0])
                    template_file = os.path.join(indir, template_stagedir, chunked_templates[i][j][0])
                else:
                    #template_file = os.path.join(outdir, template_stagedir, template_fileset_list[i][0])
                    template_file = os.path.join(outdir, template_stagedir, chunked_templates[i][j][0])
                logging.debug(f'template_file = {template_file}')
                cmd.append( template_file )            
            else:
                logging.debug(f'template_mode={template_mode}, omitting --template')

            # build full paths and check for output. 
            # build infiles/outfiles command arguments
            inlist = []
            outlist = []
            #if arity == 'parallel':
            if len(input_list) == len(output_list):
                logging.debug(f'arity=parallel output_list length={len(output_list)}')
                for k, fname in enumerate( output_list):
                    logging.debug(f'handling outfile {outdir}/{stagedir}/{fname}')
                    outfile = os.path.join(outdir, stagedir, fname)
                    if not os.path.exists(outfile):
                        outlist.append( outfile )
                        rpath = input_list[k]
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)                        
                    else:
                        logging.debug(f'outfile exists, skipping : {outfile}')

            elif (len(output_list) == 1) and (len(input_list) > 1) :
                logging.debug(f'arity=single output_list length={len(output_list)}')
                fname = output_list[0]
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    for rpath in input_list:
                        if instage is None:
                            infile = os.path.join(indir, rpath)
                        else:
                            infile = os.path.join(outdir, instage_dir, rpath)
                        inlist.append(infile)
            else:
                logging.warning(f'arity unclear. input_list len={len(input_list)} output_list len={len(output_list)}')                        

            cmd.append( '--infiles ')
            for fpath in inlist:
                cmd.append(fpath)

            cmd.append( '--outfiles ')    
            for fpath in outlist:
                cmd.append(fpath)

            if len(outlist) > 0:   
                scmd = ' '.join(cmd)
                logging.debug(f'Adding command: {scmd}')
                command_list.append(cmd)
            else:
                logging.warning(f'outlist length=0. No output files. Skip command.')

    logging.info(f'Made command list len={len(command_list)}')
    return command_list

def make_command_list_single(file_map, stage, bse, indir, outdir, cp):
    '''
    make command list
    create config
    check for output existence, skipping if all present. 
    
    '''
    cfilename = os.path.join( outdir, 'barseq.conf' )
    runconfig = write_config(cp, cfilename, timestamp=True)

    tool = cp.get( stage ,'tool')
    conda_env = cp.get( tool ,'conda_env')
    current_env = os.environ['CONDA_DEFAULT_ENV']
    instage = get_config_none(cp, stage, 'instage')
    instage_dir = None
    if instage is not None:
        instage_dir = cp.get(instage, 'stagedir')

    log_arg = ''
    log_level = logging.getLogger().getEffectiveLevel()
    if log_level <= logging.INFO:
        log_arg = '-v'
    if log_level <= logging.DEBUG : 
        log_arg = '-d'

    mode = get_config_list(cp, stage, 'modes' )
    num_cycles = int(cp.get(stage, 'num_cycles'))
    outdir = os.path.expanduser( os.path.abspath(outdir) )
    script_base = cp.get(stage, 'script_base')
    script_name = f'{script_base}_{tool}.py'
    script_dir = get_script_dir()
    script_path = f'{script_dir}/{script_name}'
    stagedir = cp.get(stage, 'stagedir')

    strip_base = cp.getboolean(stage, 'strip_base')
    template_mode = get_config_none(cp, stage, 'template_mode')
    template_source = get_config_none(cp, stage, 'template_source')

    logging.debug(f'current_env={current_env} tool={tool} conda_env={conda_env} script_dir={script_dir} script_path={script_path} script_name={script_name}')

    # Define template file(s), if requested.
    template_file_list = None
    template_stagedir = None 
    if template_mode is not None:
        template_stagedir = cp.get(template_source, 'stagedir')
        # Only tileset currently makes sense for templates. 
        template_fileset_list = bse.get_stage_files( template_mode, stage=template_source, maptype='tileset' )
        logging.debug(f'template_stagedir={template_stagedir} template_fileset_list = {template_fileset_list}')

    # Create command line(s) for mapping sets
    command_list = []
    for i, fmap in enumerate( file_map):
        logging.debug(f'handling file group {i}')
        (input_list, output_list) = fmap
        logging.debug(f'stage = {stage} file_index={i} n_input={len(input_list)} n_output={len(output_list)} num_cycles={num_cycles}')
        logging.debug(f'input = {input_list} output = {output_list}')
        cmd = []
        if conda_env != current_env:
            logging.debug(f'different envs. user conda run...')
            cmd = ['conda','run',
                        '-n', conda_env , 
                        'python', script_path,
                        log_arg, 
                        '--config' , runconfig ,                            
                        ]
        else:
            logging.debug(f'same envs needed, run direct...')
            cmd = ['python', script_path,
                        log_arg,
                        '--config' , runconfig, 
                        ]    
        cmd.append('--stage')
        cmd.append(f'{stage}')

        if template_mode is not None:
            cmd.append('--template')
            if template_source == 'input':
                template_file = os.path.join(indir, template_stagedir, template_fileset_list[i][0])
            else:
                template_file = os.path.join(outdir, template_stagedir, template_fileset_list[i][0])
            logging.debug(f'template_file = {template_file}')
            cmd.append( template_file )            
        else:
            logging.debug(f'template_mode={template_mode}, omitting --template')

        # build full paths and check for output. 
        # build infiles/outfiles command arguments
        inlist = []
        outlist = []
        #if arity == 'parallel':
        if len(input_list) == len(output_list):
            logging.debug(f'arity=parallel output_list length={len(output_list)}')
            for i, fname in enumerate( output_list):
                logging.debug(f'handling outfile {outdir}/{stagedir}/{fname}')
                outfile = os.path.join(outdir, stagedir, fname)
                if not os.path.exists(outfile):
                    outlist.append( outfile )
                    rpath = input_list[i]
                    if instage is None:
                        infile = os.path.join(indir, rpath)
                    else:
                        infile = os.path.join(outdir, instage_dir, rpath)
                    inlist.append(infile)                        
                else:
                    logging.debug(f'outfile exists, skipping : {outfile}')

        elif (len(output_list) == 1) and (len(input_list) > 1) :
            logging.debug(f'arity=single output_list length={len(output_list)}')
            fname = output_list[0]
            outfile = os.path.join(outdir, stagedir, fname)
            if not os.path.exists(outfile):
                outlist.append( outfile )
                for rpath in input_list:
                    if instage is None:
                        infile = os.path.join(indir, rpath)
                    else:
                        infile = os.path.join(outdir, instage_dir, rpath)
                    inlist.append(infile)
        else:
            logging.warning(f'arity unclear. input_list len={len(input_list)} output_list len={len(output_list)}')                        

        cmd.append('--infiles')
        for fpath in inlist:
            cmd.append(fpath)

        cmd.append('--outfiles')
        for fpath in outlist:
            cmd.append(fpath)

        if len(outlist) > 0:   
            scmd = ' '.join(cmd)
            logging.debug(f'Adding command: {scmd}')
            command_list.append(cmd)
        else:
            logging.warning(f'outlist length=0. No output files. Skip command.')

    logging.info(f'Made command list len={len(command_list)}')
    return command_list


def get_stagelist_info(cp):
    '''
    Make nicely formatted list of stages and descriptions for printing. 
    '''
    pad_w = 4
    max_w = 0
    outstring = ""

    sections = get_config_list(cp, 'experiment','stages')
    for section in sections:
        if len(section) > max_w :
            max_w = len(section)

    for section in sections:
        p_w = max_w - len(section)
        p_w += pad_w
        d = cp.get(section, 'desc')
        outstring += f"{section}{' ' * p_w}{d}\n"
    return outstring

def run_workflow(indir, outdir=None, expid=None, cp=None, halt=None):
    '''
    CSHL BARseq pipeline invocation

    Top level function to call into sub-steps...
    @arg indir          Top level input directory. Cycle directories below.  
    @arg outdir         Top-level output directory. Stage directories created below.  
    @arg cp             ConfigParser object defining stage and implementation behavior.
    @arg halt           Stop processing when halt stage is reached.  
     
    '''
    if cp is None:
        cp = get_default_config()
        
    expid = cp.get('project','project_id')
    logging.info(f'Processing experiment {expid} directory={indir} to {outdir}')
    bse = BarseqExperiment(indir, outdir, cp)
    logging.debug(f'got BarseqExperiment metadata: {bse}')
    
    # In sequence, perform all pipeline processing steps
    # maptypes are tileset, cycle, position
    try:
        stage_list = get_config_list(cp, 'experiment','stages')
        n_stages = len(stage_list)
        logging.info(f'got stage_list={stage_list}')
        for i, stage in enumerate( stage_list ):
            stage_no = i + 1
            logging.info(f'[ {stage_no}/{n_stages} ] Running stage={stage}')
            process_stage_map(indir, outdir, bse, stage=stage, cp=cp)
            logging.info(f'[ {stage_no}/{n_stages} ] Done stage={stage}')
            if stage == halt:
                logging.info(f'halt stage= {halt}, current stage {stage}, Done.')
                sys.exit(0)

        logging.info(f'Done running workflow.') 

    except Exception as ex:
        logging.error(f'got exception {ex}')
        logging.error(traceback.format_exc(None))
