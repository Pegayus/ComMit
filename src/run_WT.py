import os
import pickle

from utils import load_data
from contagion_model import ContagionModel
from test_strategy import TestStrategy

def run_WT(args):
    print(f'~~~~~~~~~~~ Running in WITH TEST mode ~~~~~~~~~')
    dirname = f'{args.mode}_{args.cmod}_{args.tmod}_None'
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
            path = os.path.join(root, file)
            print(f'############## WT: {file_name} #############')
            # check multi-value parameters
            if len(args.tbud) > 1:
                save2 = os.path.join(save2, args.id + '_var_tbud')
                if not os.path.exists(save2):
                    os.mkdir(save2)
                for tbud in args.tbud:
                    output = run(args, tbud, args.tcer[0], path, file_name)
                    with open(os.path.join(save2, str(tbud) + '.pkl'), 'wb') as f:
                        pickle.dump(output, f)
            elif len(args.tcer) > 1:
                save2 = os.path.join(save2, args.id + '_var_tcer')
                if not os.path.exists(save2):
                    os.mkdir(save2)
                for tcer in args.tcer:
                    output = run(args, args.tbud[0], tcer, path, file_name)
                    with open(os.path.join(save2, str(tcer) + '.pkl'), 'wb') as f:
                        pickle.dump(output, f)
            else:  # both are single-value
                output = run(args, args.tbud[0], args.tcer[0], path, file_name)
                with open(os.path.join(save2, args.id + '.pkl'), 'wb') as f:
                    pickle.dump(output, f)



def run(args, tbud, tcer, path, file_name):      
    G_u0, G_k0, sources, _ = load_data(path)
    # {args: , graph_u: , hist: [{shist1: ,thist1},..., {shist10, thist10:}] }
    output = {'args': vars(args).copy(), 'graph_u': G_u0.copy(), 'hist': []}
    # loop over sources
    count = 0
    for source in sources:
        count += 1
        G_u = G_u0.copy()
        G_k = G_k0.copy()
        # spread init
        states = {k:'S' for k in G_u.nodes()}
        states.update({k:'I' for k in source})
        spread = ContagionModel(graph = G_u, states = states, model = args.cmod,
                                    duration_infectious=args.doi, infection_rate = args.ir)
        # test init
        params = dict(visited=[], epsilon=args.eps, rec_pos=[], decay_factor=args.df)
        budget = int(tbud * G_k.number_of_nodes())
        test = TestStrategy(method = args.tmod, graph_unknown = G_u, spread_model = spread, graph_known = G_k, 
                            test_budget = budget, trace_acc = tcer, **params)
        for _ in range(args.sd):
            test.run()
            spread.run()
            if spread.terminate:
                break
        # add results to output 
        res = {'shist': spread.get_history(), 'thist': test.get_history()}
        output['hist'].append(res)
        print(f'WT_tbud{tbud}_tcer{tcer}_{file_name}: Source set {count} done.')
    return output
            
