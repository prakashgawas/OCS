#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=300M
#SBATCH --time=00:20:00
#SBATCH --partition=optimum
#SBATCH --array=1-700
#SBATCH --output=../Doutput/arrayjob_%A_%a.out


module load gurobi/11.0.0
module load python
module load anaconda
conda activate scheduler_env

i=1
P=4

for n in  {6701..7000}
do
        	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		python ../src/OCS.py --seed $n --N1 25 --s1 2.5 --N2 75 --s2 6 --P $P --I 8 --L 20 --var 0 --kmin 1 --kmax $P --pw 1 --time_limit 60 --cnp 10 --co 0 > ../Doutput/output_$n$s.txt
		fi
	    	(( i = $i +1 ))
done

sleep 60