import os
import numpy as np

from generate_data import generate_hcvrp_data
from my_utils.data_utils import check_extension, save_dataset
from my_utils.generate_mtw import generate_time_windows
import torch
import pickle
import argparse

def generate_hcvrptw_data(dataset_size, hcvrptw_size, veh_num):
    data = []
    for seed in range(24601, 24611):
        rnd = np.random.RandomState(seed)

        loc_lat = rnd.uniform(45.705530, 45.812706, size=(dataset_size, hcvrptw_size * 2))
        lonc_long = rnd.uniform(4.764332, 4.924109, size=(dataset_size, hcvrptw_size * 2))
        cust = np.column_stack((loc_lat[:, :hcvrptw_size], lonc_long[:, :hcvrptw_size]))
        pickup = np.column_stack((loc_lat[:, hcvrptw_size:], lonc_long[:, hcvrptw_size:]))

        d = rnd.randint(1, 10, [dataset_size, hcvrptw_size * 2])

        # Generate random service times for each customer (pickup and delivery)
        service_times = rnd.randint(1, 21, size=(dataset_size, hcvrptw_size * 2))

        # Set score to 100 for each customer
        scores = rnd.randint(100, 100, size=(dataset_size, hcvrptw_size * 2))

        # Generate random time windows for each customer (pickup and delivery)
        time_windows = generate_time_windows(dataset_size * hcvrptw_size*2)
        # time_windows_lower = rnd.randint(420, 1200 - 60, size=(dataset_size, hcvrptw_size * 2))
        # time_windows_upper = rnd.randint(time_windows_lower + 60, 1200, size=(dataset_size, hcvrptw_size * 2))

        if veh_num == 3:
            cap = [100., 200., 300.]
            thedata = list(zip(cust.tolist(),
                               pickup.tolist(),
                               d.tolist(),
                               service_times.tolist(),
                               scores.tolist(),
                               np.full((dataset_size, 3), cap).tolist(),
                               time_windows.toliste()
                               ))
            data.append(thedata)
        elif veh_num == 2:
            cap = [100., 250., 0.]
            thedata = list(zip(cust.tolist(),
                               pickup.tolist(),
                               d.tolist(),
                               service_times.tolist(),
                               scores.tolist(),
                               np.full((dataset_size, 3), cap).tolist(),
                               time_windows.toliste()
                               ))
            data.append(thedata)
        elif veh_num == 1:
            cap = [100., 0., 0.]
            thedata = list(zip(cust.tolist(),
                               pickup.tolist(),
                               d.tolist(),
                               service_times.tolist(),
                               scores.tolist(),
                               np.full((dataset_size, 3), cap).tolist(),
                               time_windows.toliste()
                               ))
            data.append(thedata)

    data = np.array(data).reshape(dataset_size * 10, 7)
    return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=128, help="1/10 Size of the dataset")
    parser.add_argument("--veh_num", type=int, default=3, help="number of the vehicles; 3 or 5")
    parser.add_argument('--graph_size', type=int, default=40,
                        help="Sizes of problem instances: {40, 60, 80, 100, 120} for 3 vehicles, "
                             "{80, 100, 120, 140, 160} for 5 vehicles")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'hcvrp'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating HCVRP data
    np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size)
    filename = os.path.join(datadir, '{}_v{}_{}_seed{}.pkl'.format(problem, opts.veh_num, opts.graph_size, seed))

    dataset = generate_hcvrp_data(opts.dataset_size, opts.graph_size, opts.veh_num)
    print(dataset[0])
    save_dataset(dataset, filename)
