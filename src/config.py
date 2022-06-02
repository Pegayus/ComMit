'''
NOTE
Multi-value parameters:
    - tbud
    - tcer
    - mcbud
    - mnbud
The run.py is written with the assumption that at each run only one of
these parameters is multi-value (i.e., len(args.<parameter>) > 1).
If you wish to have several of them as multi-value, you have to change run.py
accordingly.
'''

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    # -------- handling paths and mode of running
    parser.add_argument('--dpath', type=str, default='../data', 
                            help='The directory to read the data from sequentially.')
    parser.add_argument('--spath', type=str, default='../result', 
                            help='The directory to save the results in.')
    parser.add_argument('--id', type=str, required=True, 
                            help='Identifier used as the name that the file is saved as \
                                  in this format: spath/data_name/id')
    parser.add_argument('--mode', type=str, default='WI', choices=['NI', 'WT', 'WI'], 
                            help='The mode of running the code. NI: no intervestion (e.g. just SIR) || \
                                     WT: with test || WI: with intervention (test + mitigation)')
    parser.add_argument('--sd', type=int, default=100, help='Simulation duration.')

    # --------- contagion (spread) prameters
    parser.add_argument('--cmod', type=str, default='SIS', choices=['SIR', 'SIS'],
                            help='Congation model for spread, SIR or SIS are supported.')
    parser.add_argument('--ir', type=float, default=0.5, 
                            help='Infection rate for SIR or SIS contagion model.')
    parser.add_argument('--doi', type=int, default=3, 
                            help='Duration of infection for SIR or SIS contagion model.')

    # -------- test parameters
    parser.add_argument('--tmod', type=str, default='random', 
                            help='Test model to use as testing strategy. \
                                  Options: random, random_with_memory, random_with_selective_memory, \
                                    epsiolon_greedy, epsilon_memory')
    parser.add_argument('--tcer', type=float, default=[1.], nargs='+', 
                            help='List of trace certainties in testing model. \
                                  Trace certainly: portion of neighbors that are self-reported.\
                                  Values are in (0,1] range')
    parser.add_argument('--tbud', type=float, default=[0.1], nargs='+', 
                            help='List of test budgets in terms of the proportion of \
                                  network nodes (e.g., 0.1 means 0.1*num_nodes. \
                                  Values are in (0,1) range.')
    # test specific parameters
    parser.add_argument('--eps', type=float, default=0.99, 
                            help='Epsilon value for epsilon-based tests.')
    parser.add_argument('--df', type=int, default=2,
                            help='Decay factor for epsilon-based tests.')

    # -------- mitigation parameters
    parser.add_argument('--mmod', type=str, default='commit', 
                            choices=['commit', 'comiso', 'degiso', '1hopiso'],
                            help='Mitigation strategy to be used with testing startegy. \
                                  commit: community fragmentation-based || \
                                  comiso: community isolation based on threshold || \
                                  degiso: degree-based isolation of groups (high degree node chosen as candidate) \
                                  1hopiso: isolating 1hop neighborhood of known infected.')
    parser.add_argument('--mcbud', type=float, default=[0.01], nargs='+',
                            help='Mitigation candidate budget: how many candidates to pick \
                                  each round as the center of fragmentation, in terms of \
                                  the proportion of the nodes in the network.')
    parser.add_argument('--mnbud', type=int, default=[4], nargs='+',
                            help='Mitigation neighbor selection budget: how many neighbors \
                                  the candidate will be restricted with.')
    parser.add_argument('--mrd', type=int, default=4, 
                            help='Mitigation restriction duration.')
    # mitigation specific parameters
    parser.add_argument('--cthr', type=float, default=0.1,
                            help='Threshold for comiso method in terms of the proportion of \
                                  the community size. The values are in (0,1) range.')

    args = parser.parse_args()
    if not os.path.exists(args.spath):
        os.makedirs(args.spath)
    return args