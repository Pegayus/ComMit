import os
import pickle

from utils import load_data
from contagion_model import ContagionModel
from test_strategy import TestStrategy
from mitigation_strategy import MitigationStrategy


def run_WI(args):
    print(f'~~~~~~~~~~~ Running in WITH INTERVENTION mode ~~~~~~~~~')
    dirname = f'{args.mode}_{args.cmod}_{args.tmod}_{args.mmod}'
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
            print(f'############## WI: {file_name} #############')
            # check multi-value parameters
            if len(args.mcbud) > 1:
                save2 = os.path.join(save2, args.id + '_var_mcbud')
                if not os.path.exists(save2):
                    os.mkdir(save2)
                for mcbud in args.mcbud:
                    output = run(args, mcbud, args.mnbud[0], path, file_name)
                    with open(os.path.join(save2, str(mcbud) + '.pkl'), 'wb') as f:
                        pickle.dump(output, f)
            elif len(args.mnbud) > 1:
                save2 = os.path.join(save2, args.id + '_var_mnbud')
                if not os.path.exists(save2):
                    os.mkdir(save2)
                for mnbud in args.mnbud:
                    output = run(args, args.mcbud[0], mnbud, path, file_name)
                    with open(os.path.join(save2, str(mnbud) + '.pkl'), 'wb') as f:
                        pickle.dump(output, f)
            else:  # both are single-value
                output = run(args, args.mcbud[0], args.mnbud[0], path, file_name)
                with open(os.path.join(save2, args.id + '.pkl'), 'wb') as f:
                    pickle.dump(output, f)



def run(args, mcbud, mnbud, path, file_name):      
    G_u0, G_k0, sources, clusters = load_data(path)
    # {args: , graph_u: , hist: [{shist1: ,thist1:, mhist1:},..., {shist10, thist10:,, mhist10:}] }
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
        budget = int(args.tbud[0] * G_k.number_of_nodes())
        test = TestStrategy(method = args.tmod, graph_unknown = G_u, spread_model = spread, graph_known = G_k, 
                            test_budget = budget, trace_acc = args.tcer[0], **params)
        # mitigation init
        params = dict(restrict_time=args.mrd, restrict_candidate_budget=int(mcbud * G_k.number_of_nodes()), 
                        restrict_candidate_neigh_budget=mnbud, community_thr=args.cthr)
        mitigate = MitigationStrategy(method = args.mmod, graph_known = G_k, graph_unknown = G_u,
                                      clusters = clusters, test_states = test.get_states(), **params)
        for _ in range(args.sd):
            # test
            test.run()
            # mitigate
            mitigate.set_graph_k(test.get_graph_k())
            mitigate.run(test.get_states(), test.get_latest_inf())
            # spread
            spread.set_graph(mitigate.get_graph_u())
            spread.run()
            if spread.terminate:
                break
        # add results to output 
        res = {'shist': spread.get_history(), 'thist': test.get_history(), 'mhist': mitigate.get_history()}
        output['hist'].append(res)
        print(f'WI_mcbud{mcbud}_mnbud{mnbud}_{file_name}: Source set {count} done.')
    return output
            
