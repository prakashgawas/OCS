#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=800M
#SBATCH --time=00:20:00
#SBATCH --partition=optimum
#SBATCH --array=1-20
#SBATCH --output=../Moutput/arrayjob_%A_%a.out
#SBATCH --begin=now+0hour

i=1
w=0
loc="server"
P=4
li=400
cont=0
um=1
lm=0.8
s=8
vanilla=1 #change
gated=0 #change
tl=30
sims=10
stoch=0
adapter="flat"
mg=0.02
	for kmin in 1 
do
		for runs in  10 #50 200 500 1000 1500 2000 2500 3000 3500 4000
	do
		for pw in 1 
	do

	for scen in 30
	do
		for new in 1 
	do
		for stoch in 1 
		do
			for lm in 0 0.5 1
			do

	        if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    		    	python ../src/Merge_sim_trace.py --kmin $kmin --s $s --adapter $adapter --pw $pw --new $new  --scenarios $scen --stoch $stoch --lm $lm --learn_iter $li --runs $runs --time_limit $tl --evolve_expert 0 --mipgap $mg > ../Moutput/Moutput$kmin$new$mg$scen.txt
		fi
	    	(( i = $i +1 ))
	done
done
done
done
done
done

done


sleep 6
