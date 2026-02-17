#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=00:50:00
#SBATCH --array=1-1000
#SBATCH --output=output/arrayjob_%A_%a.out
#SBATCH --partition=optimum

module load gurobi
#source ~/env_gurobi/bin/activate

i=1
for n in  {1..1000}
do
    for scen in 20 
    do

        if [ $SLURM_ARRAY_TASK_ID -eq $i ]
        then
             python SONCSS_Main.py --scen $scen  --seed $n --tl 45 > output/outputstoch_$n$scen.txt
        fi
        (( i = $i +1 ))
    done
done

sleep 60