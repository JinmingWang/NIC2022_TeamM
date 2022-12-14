# NIC2022_TeamM

## Team members (Names in alphabetical order):
- Brian Pereira
- Duanyu Lee
- Jinming Wang
- Shujing Sun
- Tianyu Gong

## Problem: Worst 1-max solver (GECCO 2007)
- Introdution URL: https://www.sigevo.org/gecco-2007/competitions.html#c2
- One-Max problem: Find the maximum sum of a length N bit string (a string consisting of only 1s and 0s)
- Conatraints
    1. Algorithm should solve the problem with above 95% chance (find the optimal bitstring 95 times out of 100 runs)
    2. Solve the problem as late as possible, within 1000 generations (for example, solves the problem in 100 generation is not good, solves it in 700 generations is good)
    3. At most 10,0000 fitness evaluations can be made.
    4. Parameters can change during EA running, but only linearly.

## How to Run
- OneMaxObject.py
  - When you run it, it does some testing to Population and Bitstring
- SampleOneMaxSolver.py
  ```console
  $py SampleOneMaxSolver.py [-h] [-p POP_SIZE] [-m MUTATION] [-r] [-e] [-s]
  ```
  - -h is help
  - -p defines the population size
  - -m defines the mutation number
  - -r will run algorithm once
  - -e will run algorithm 100 times
  - -s will run parameters searching
  - You should define one of -r, -e or -s
- LinearDecayMutationSolver.py
  ```console
  $py LinearDecayMutationSolver.py [-h] [--pop_size POP_SIZE] [--init_mutation INIT_MUTATION] [--mutation_decay MUTATION_DECAY] [--min_mutation MIN_MUTATION] [-r] [-e] [-s] [-a]
  ```
  - h is help
  - --pop_size defines the population size
  - --init_mutation defines the initial mutation rate
  - --mutation_decay defines the decay of mutation rate on teach generation
  - --min_mutation define the minimum mutation rate
  - -r will run algorithm once
  - -e will run algorithm 100 times and evaluate
  - -s will run parameters searching and print result
  - -a will run a filtering and selecting of searching result
  - You should define one of -r, -e, -s or -a
- PowerfulSolver.py
  - There are many versions of PowerfulSolver, since everyone was implementing different experiments
  - Running Each PowerfulSolver directly should be able to produce the experiment results
- EAofEA/EAEA.py
  ```console
  $py EAEA.py [-h] [--pop_size POP_SIZE] [--t_size T_SIZE]
               [--t_select T_SELECT] [--mutation_factor MUTATION_FACTOR]
               [--generations GENERATIONS] [--load_file LOAD_FILE]
               [--processes PROCESSES] [--n_trials N_TRIALS] [-r] [-e]
  ```
  - -h is help
  - --pop_size is population size
  - --t_size is tournament size
  - --t_select is number of parent selected
  - --mutation_factor is the magnitude of mutation
  - --generations is number of generations to run EAofEA
  - --load_file: if to load a saved checkpoint, what is the path
  - --processes is number of processes to run the code
  - --n_trials is number of trials to evaluate each parameter combination when evaluating
  - -r is to run EAofEA
  - -e is to evaluate EAofEA
  - You should define one or more of -r or -e
