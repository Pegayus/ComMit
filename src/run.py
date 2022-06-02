from config import get_args
from run_NI import run_NI
from run_WT import run_WT
from run_WI import run_WI
# from utils import calc_spread_metric, calc_test_metric, calc_mit_metric, load_data


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'NI':
        run_NI(args)
    elif args.mode == 'WT':
        run_WT(args)
    else:
        run_WI(args)

