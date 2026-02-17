#!/bin/bash
#SBATCH --cpus-per-task=2    # 2 threads for each worker
#SBATCH --ntasks=10          # 10 workers
#SBATCH --mem=8G
#SBATCH --time=147:00:00
#SBATCH --partition=optimumlong
#SBATCH --array=1-4
#SBATCH --output=../Loutput/arrayjob_%A_%a.out
#SBATCH --begin=now+0hour

module load gurobi/11.0.0

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
mg=0.0001
s=8
adapter="flat"
new=1
#check stoch li runs new scen
for N in 100
do
for runs in 10
	do
for scen in  30
	do
	#for mg in  0.02
#		do
	for stoch in  1
		do
	for pw in  1
		do
	for kmin in 1 
		do
			for tl in 5 10 20
			do

        	if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		python ../src/Learn_Main.py  --N $N --s $s --P $P --I 8 --L 20 --adapter $adapter  --kmin $kmin --kmax $P --pw $pw  --cnp 10 --co 0  --runs $runs --new $new  --scenarios $scen --stoch $stoch --um $um --learn_iter $li  --resume $cont --use_weight $w --lm $lm --time_limit $tl --vanilla $vanilla --gated $gated --evolve_expert 0 --mipgap $mg > ../Loutput/output$li$P$s$tl$kmin$pw$scen$new$stoch$runs.txt	
	    		
		fi
	    	(( i = $i +1 ))
done
done
#done
done
done
done
done
done

sleep 60