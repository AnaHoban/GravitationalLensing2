#!/bin/bash

#SBATCH --account=def-sfabbro
#SBATCH --time=0-2:00

## create 100 jobs
#SBATCH --array=0-99
#SBATCH --mem=8000M
#SBATCH --output=outputs/%x-%j.out

# a small function that will call your python script to run on one tile
function infer_one_tile() {
    local tile=$1
    local bands_code = $2

    echo "downloading ${tile} files on host $(hostname)"
    date
    
    # where to download each tile
    local workdir=${SLURM_TMPDIR}/${tile}
    mkdir -p ${workdir}
    
    if [[ $#bands == 1 ]] || [ $#bands == 3 ]]; then
        vcp -v vos:cfis/tiles_DR3/${tile}.u.fits ${workdir}/
        vcp -v vos:cfis/tiles_DR3/${tile}.u.cat ${workdir}/
    else if [[ $#bands == 2]] || [[ $#bands == 3 ]]; then
        vcp -v vos:cfis/tiles_DR3/${tile}.r.fits ${workdir}/
        vcp -v vos:cfis/tiles_DR3/${tile}.r.cat ${workdir}/
    fi
    
    
    # PanSTARRS tiles (you may ignore this for now)
    #if [[ ${tile} =~ PS1 ]]; then
    #    vcp -L -v vos:cfis/ps_tiles/${tile}* ${workdir}/
    # CFIS tiles

    echo "performing inference on ${tile}"
    date

    python inference_pipeline.py ${tile}

    #echo "saving outputs"
    #cp ${workdir} ${SCRATCH}/my_output.csv

    # cleanup, ready for next tile
    rm -fv ${workdir}
}


source $HOME/umap/bin/activate

# create a file 'tile.list' with list of all available tiles:
# 0. Get a certificate:

#   cadc-get-cert -u AnaHoban

# 1. List all files in the tiles_DR3 directory

#   vls vos:cfis/tiles_DR3 > tiles_DR3.vls

# 2. From the list of all files, create a file tile.list with unique tile names

#   cat tiles_DR3.vls | sed -e 's|\(CFIS........\).*|\1|g' | sort | uniq > tile.list


# create an array of all the tiles
tile_list=($(<all_files.list))


# set the number of tiles that each SLURM task should do
per_task=1000

# starting and ending indices for this task
# based on the SLURM task and the number of tiles per task.
start_index=$(( (${SLURM_ARRAY_TASK_ID} - 1) * ${per_task} + 1 ))
end_index=$(( ${SLURM_ARRAY_TASK_ID} * ${per_task} ))


echo "This is task ${SLURM_ARRAY_TASK_ID}, which will do tiles ${tile_list[${start_index_unique}]} to ${tile_list[${end_index_unique}]}"

for (( idx=${start_index}; idx<=${end_index}; idx++ )); do
    tile=${tile_list[${idx}]}
    next_tile=${tile_list[${idx}]}
    echo "This is SLURM task ${SLURM_ARRAY_TASK_ID} for tile ${tile}"
    bands = 1 #default --just u

    if [[ ${tile} == ${next_tile} ]]; then
        bands = 3 #we have both bands
        idx ++ #skip the next
    elif [[ ${tile}[-1] == 'r' ]]; then
         bands = 2
    fi
    
    infer_one_tile ${tile} ${bands_nb}
done