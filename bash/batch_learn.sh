#!/bin/bash
#SBATCH --cpus-per-task=2    # 2 threads for each worker
#SBATCH --ntasks=10          # 10 workers
#SBATCH --mem=10G
#SBATCH --time=47:00:00
#SBATCH --partition=optimum
#SBATCH --array=1-1
#SBATCH --output=../Loutput/arrayjob_%A_%a.out
#SBATCH --begin=now+0hour

module load gurobi/11.0.0
module load anaconda
conda activate scheduler_env

N1=25
N2=75
s1=5
s2=12
var=1
i=1
w=0
loc="server"
P=4
li=400
cont=0
um=1
lm=0.8
vanilla=1 #change
gated=0 #change
tl=30
mg=0.001
adapter="flat"
pw=1
#check stoch li runs new scen

for runs in 10
	do
for scen in 30 
	do
	for mg in  0.01 #0.01 # 0.05 0.1
		do
	for stoch in  1
		do
	for lm in  0.8
		do
	for kmin in 1 
		do
			for new in  1 
			do

        	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		python ../src/Learn_Main.py  --N1 $N1 --s1 $s1 --N2 $N2 --s2 $s2 --P $P --I 8 --L 20 --adapter $adapter --var $var --kmin $kmin --kmax $P --pw $pw  --cnp 10 --co 0  --runs $runs --new $new  --scenarios $scen --stoch $stoch --um $um --learn_iter $li  --resume $cont --use_weight $w --lm $lm --time_limit $tl --vanilla $vanilla --gated $gated --evolve_expert 0 --mipgap $mg > ../Loutput/output$li$P$s1$tl$mg$scen$new$stoch$runs.txt	
	    		
		fi
	    	(( i = $i +1 ))
done
done
done
done
done
done
done

sleep 60