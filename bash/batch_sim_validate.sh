#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --partition=optimum
#SBATCH --array=1-810
#SBATCH --output=../Soutput/arrayjob_%A_%a.out
#SBATCH --begin=now+0hour

module load gurobi/11.0.0
module load python
module load anaconda
conda activate scheduler_env

i=1
w=0
loc="server"
P=4
li=400
cont=0
um=1
lm=0.8
stoch=1
vanilla=1 #change
#runs=1500
gated=0 #change
tl=30
sims=1000
N1=25
N2=75
s1=5
s2=12
var=1
pw=1
mg=0.02
kmin=1
adapter="flat"
#check stoch li runs new  scen
for k in  {0..400}
do   
	for runs in  10 #50 200 500 1000 1500 2000 2500 3000 3500 4000 #10 #2000 #
	do
	for mg in 0.01 
do
	for lm in 0.8
	do
	for stoch in  1
		do
	for scen in    30
	do
		for new in 1
		do

	        if [ $SLURM_ARRAY_TASK_ID -eq $i ]
		then
	    	python ../src/Learn_Sim.py --sims $sims --seed 6001  --model $k --store_sim 0 --var $var --N1 $N1 --s1 $s1 --N2 $N2 --s2 $s2 --adapter $adapter  --P $P --I 8 --L 20 --kmin $kmin --kmax $P --pw $pw  --cnp 10 --co 0  --runs $runs  --scenarios $scen --stoch $stoch --um $um --learn_iter $li  --resume $cont --use_weight $w --lm $lm --time_limit $tl --new $new --vanilla $vanilla --gated $gated --evolve_expert 0 --mipgap $mg > ../Soutput/Soutput$mg$kmin$runs$new$s1$pw$P$k$tl$scen$stoch.txt	
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
