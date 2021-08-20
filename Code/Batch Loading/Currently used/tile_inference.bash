#!/bin/bash

#SBATCH --account=def-sfabbro
#SBATCH --time=4-15:00

## create 100 jobs
#SBATCH --array=0-199
#SBATCH --mem=8000M
#SBATCH --output=outputs/%x-%j.out


source $HOME/umap/bin/activate

# a small function that will call your python script to run on one tile
function infer_one_tile() {
    local tile=$1
    local bands_nb=$2

    echo "downloading ${tile} files on host $(hostname)"
    date
    
    # where to download each tile
    local workdir=${SLURM_TMPDIR}/${tile}
    mkdir -p ${workdir}
    
    if [[ ${bands_nb} -eq 1 ]] || [[ ${bands_nb} -eq 3 ]]; then
        vcp -v vos:cfis/tiles_DR3/${tile}.u.fits ${workdir}/
        vcp -v vos:cfis/tiles_DR3/${tile}.u.cat ${workdir}/
    fi
    if [[ ${bands_nb} -eq 2 ]] || [[ ${bands_nb} -eq 3 ]]; then
        vcp -v vos:cfis/tiles_DR3/${tile}.r.fits ${workdir}/
        vcp -v vos:cfis/tiles_DR3/${tile}.r.cat ${workdir}/
    fi

    echo "performing inference on ${tile}"
    date

    python inference_pipeline.py ${tile} ${workdir}

    # cleanup, ready for next tile
    rm -rf ${workdir}
}


tile_list=($(<remaining_tiles.list))

# set the number of tiles that each SLURM task should do (43 031 tiles in total)
per_task=216

# starting and ending indices for this task
# based on the SLURM task and the number of tiles per task.
start_index=$(( (${SLURM_ARRAY_TASK_ID} - 1 ) * ${per_task} + 1 ))
end_index=$(( ${SLURM_ARRAY_TASK_ID} * ${per_task} ))

echo "This is task ${SLURM_ARRAY_TASK_ID}, which will do tiles ${tile_list[${start_index}]} to ${tile_list[${end_index}]}"

idx=${start_index}

while (( ${idx} <= ${end_index} )); do
    
    tile=${tile_list[${idx}]}
    next_tile=${tile_list[${idx}+1]}
    
    if ((${idx}+1 > ${end_index})); then
        next_tile=1 #no match
    fi    
    
    echo "This is SLURM task ${SLURM_ARRAY_TASK_ID} for tile ${tile[@]/${tile: -2}}"
    
    bands=1 #default --just u
  
    #checking if the next tile has the same tile ID
    if [[ ${tile[@]/${tile: -1}} == ${next_tile[@]/${next_tile: -1}} ]]; then
    
        bands=3 #we have both bands
        idx=$((${idx}+1)) #skip the next tile name 
        
    elif [[ ${tile: -1} == 'r' ]]; then
         bands=2
    fi    
    
    echo "${bands} band code"
    infer_one_tile ${tile[@]/${tile: -2}} ${bands}
    
    idx=$((${idx}+1)) 
done