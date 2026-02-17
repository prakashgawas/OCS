# Problem Description: Dynamic Physician-to-Patient Assignment (PPA)

## Overview
We consider a **dynamic physician-to-patient assignment (PPA)** problem arising in clinic appointment booking. Patients (call-ins) arrive sequentially over a booking horizon until a cutoff time for a particular session. The clinic is staffed by multiple physicians, and each incoming request must be handled immediately by a receptionist.

At each call-in, the receptionist must decide to **accept** or **reject** the request. If accepted, the receptionist must also decide **which physician** to assign to the patient among the patient’s eligible physicians. Decisions must be made online under uncertainty about future arrivals.

## Physicians
Let $\mathcal{P} = \{1, \dots, P\}$ denote the set of available physicians. Each physician $p \in \mathcal{P}$ has:
- a regular session time $T$ (in minutes),
- a maximum appointment capacity $L_p$.

## Patients (Call-ins)
Each patient request $n$ is characterized by:
- an eligible physician set $\mathcal{P}_n \subseteq \mathcal{P}$,
- a preferred physician $p_n \in \mathcal{P}_n$,
- an estimated service duration $d_n \in \mathbb{R}_{+}$ (known at call-in time),
- a priority class $r_n \in \{1,2\}$, where:
  - class 1: high priority,
  - class 2: regular priority.

## Costs / Penalties
If the receptionist rejects patient $n$, a rejection penalty is incurred:
$c^{\text{pref}}_{r_n}$.

If the receptionist accepts patient $n$ but assigns them to an eligible
non-preferred physician, a preference penalty is incurred:
$c^{\text{pref}}_{r_n}$, where $r_n \in \{1,2\}$.

## Key Challenge (Uncertainty and Trade-offs)
The number and composition of future requests are uncertain. This creates a trade-off between:
- accepting lower-priority patients now versus saving capacity for potentially higher-priority patients later,
- assigning a patient to a non-preferred physician (incurring a penalty) versus rejecting them (incurring a potentially larger penalty),
- preserving feasibility for future patients who may have limited eligibility, since poor early assignments can force future rejections.

## Decision Process View
This problem can be modeled as a **stochastic dynamic program (SDP)**:
- each decision epoch corresponds to an arriving patient call-in,
- the system state includes remaining physician capacities (and implicitly the remaining booking horizon),
- actions are to assign the patient to a physician $p \in \mathcal{P}_n$ or to reject the patient.

The objective is to choose decisions over the booking horizon to **minimize the total expected penalties** from patient rejections and non-preferred assignments, subject to physician capacity constraints.


# Running the Code to learn policies
python Learn_Main.py  --N 100 --s 8 --P 4 --I 8 --L 20  --kmin 1 --kmax 4 --pw 1  --cnp 10 --co 0  --runs 10 --new 1  --scenarios 5 --stoch 1 --um 1 --learn_iter 100   --time_limit 20  --mipgap 0.02 
# Running the Code to simulate learnt policies
python Learn_Sim.py --sims 1000 --model 1 --store_sim 0 --N 100 --s 8 --P 4 --I 8 --L 20 --kmin 1 --kmax 4 --pw 1  --cnp 10 --co 0  --runs 10 --new 1  --scenarios 5 --stoch 1 --um 1 --learn_iter 100  --time_limit 20 --mipgap 0.02 


