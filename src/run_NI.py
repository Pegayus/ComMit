import os
import pickle

from utils import load_data
from contagion_model import ContagionModel


def run_NI(args):
    print(f'~~~~~~~~~~~ Running in NO INTERVENTION mode ~~~~~~~~~')
    dirname = f'{args.mode}_{args.cmod}_None_None'
    save2 = os.path.join(args.spath, dirname)
    if not os.path.exists(save2):
        os.makedirs(save2)
    # loop over data
    for root, _, files in os.walk(args.dpath):
        for file in files:
            file_name = file.split('.pkl')[0]
            save2 = os.path.join(save2, file_name)
            if not os.path.exists(save2):
                os.mkdir(save2)
            print(f'############## NI: {file_name} #############')
            G_u0, _, sources, _ = load_data(os.path.join(root, file))
            # {args: , graph_u: , hist: [{thist1: }, {thist2: },..., {thist3:}] }
            output = {'args': vars(args).copy(), 'graph_u': G_u0.copy(), 'hist': []}
            # loop over sources
            count = 0
            for source in sources:
                count += 1
                G_u = G_u0.copy()
                # initialization
                states = {k:'S' for k in G_u.nodes()}
                states.update({k:'I' for k in source})
                spread = ContagionModel(graph = G_u, states = states, model = args.cmod,
                                            duration_infectious=args.doi, infection_rate = args.ir)
                for _ in range(args.sd):
                    spread.run()
                    if spread.terminate:
                        break
                # add results to output 
                res = {'shist': spread.get_history()}
                output['hist'].append(res)
                print(f'NI_spread_{file_name}: Source set {count} done.')
            # save output under dirname/data/id.pkl
            with open(os.path.join(save2, args.id + '.pkl'), 'wb') as f:
                pickle.dump(output, f)
