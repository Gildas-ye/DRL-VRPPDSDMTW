

from torch.utils.data import Dataset
import torch
import os
import pickle
from problem.hcvrp.state_hcvrp import StateHCVRP
from my_utils.beam_search import beam_search


class HCVRP(object):
    NAME = 'hcvrp'

    VEHICLE_CAPACITY = [100., 200., 300.]

    @staticmethod
    def get_costs(dataset, obj, pi, veh_list, tour_1, tour_2, tour_3):
        SPEED = [1/4, 1/5, 1/6]
        batch_size, graph_size = dataset['demand'].size()
        num_veh = len(HCVRP.VEHICLE_CAPACITY)

        # # Check that tours are valid, i.e. contain 0 to n -1, [batch_size, num_veh, tour_len]
        sorted_pi = pi.data.sort(1)[0]
        # Sorting it should give all zeros at front and then 1...n
        assert (torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                sorted_pi[:, -graph_size:]
                ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        demand_with_depot = torch.cat(  # [batch_size, graph_size]
            (
                torch.full_like(dataset['demand'][:, :1], 0),  # pickup problem, set depot demand to -capacity
                dataset['demand']
            ),
            1
        )
        # pi: [batch_size, tour_len]
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0:num_veh])  # batch_size, 3
        # for veh in range(num_veh):  # num_veha
        for i in range(pi.size(-1)):  # tour_len
            # print('d', i, d[0, i])
            used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] += d[:,
                                                                                         i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            used_cap[used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] < 0] = 0
            used_cap[(tour_1[:, i] == 0), 0] = 0
            assert (used_cap[torch.arange(batch_size), 0] <=
                    HCVRP.VEHICLE_CAPACITY[0] + 1e-5).all(), "Used more than capacity 1"
            used_cap[(tour_2[:, i] == 0), 1] = 0
            assert (used_cap[torch.arange(batch_size), 1] <=
                    HCVRP.VEHICLE_CAPACITY[1] + 1e-5).all(), "Used more than capacity 2"
            used_cap[(tour_3[:, i] == 0), 2] = 0
            assert (used_cap[torch.arange(batch_size), 2] <=
                    HCVRP.VEHICLE_CAPACITY[2] + 1e-5).all(), "Used more than capacity 3"

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # batch_size, graph_size+1, 2

        # [batch_size, tour_len, 2]
        dis_1 = loc_with_depot.gather(1, tour_1[..., None].expand(*tour_1.size(), loc_with_depot.size(-1)))
        dis_2 = loc_with_depot.gather(1, tour_2[..., None].expand(*tour_2.size(), loc_with_depot.size(-1)))
        dis_3 = loc_with_depot.gather(1, tour_3[..., None].expand(*tour_3.size(), loc_with_depot.size(-1)))

        total_dis_1 = (((dis_1[:, 1:] - dis_1[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_1[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_1[:, -1] - dataset['depot']).norm(p=2, dim=1)) / SPEED[0]).unsqueeze(
            -1)  # [batch_size]
        total_dis_2 = (((dis_2[:, 1:] - dis_2[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_2[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_2[:, -1] - dataset['depot']).norm(p=2, dim=1)) / SPEED[1]).unsqueeze(
            -1)  # [batch_size]
        total_dis_3 = (((dis_3[:, 1:] - dis_3[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_3[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_3[:, -1] - dataset['depot']).norm(p=2, dim=1)) / SPEED[2]).unsqueeze(
            -1)  # [batch_size]

        total_dis = torch.cat((total_dis_1, total_dis_2, total_dis_3), -1)
        if obj == 'min-max':
            return torch.max(total_dis, dim=1)[0], None
        if obj == 'min-sum':
            return torch.sum(total_dis, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = HCVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float)
    }


class HCVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: [20., 25., 30.],
                20: [20., 25., 30.],
                40: [20., 25., 30.],
                50: [20., 25., 30.],
                60: [20., 25., 30.],
                80: [20., 25., 30.],
                100: [20., 25., 30.],
                120: [20., 25., 30.],
            }
            # capa = torch.zeros((size, CAPACITIES[size]))

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float(),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'capacity': torch.Tensor(CAPACITIES[size])
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data


