#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=200M
#SBATCH --time=00:30:00
#SBATCH --partition=optimum
#SBATCH --array=1-1000
#SBATCH --output=../output/arrayjob_%A_%a.out

module load gurobi/11.0.0

i=1
P=4
for n in  {1000..1500}
do
        	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		python ../src/OCS.py --seed $n --N 100 --s 8 --P $P --I 8 --L 20 --kmin 1 --kmax $P --pw 1 --time_limit 60 --cnp 10 --co 0 > ../output/output_$n$s.txt
		fi
	    	(( i = $i +1 ))
done

sleep 60