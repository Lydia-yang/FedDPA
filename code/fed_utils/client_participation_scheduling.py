import numpy as np


def client_selection(num_clients, client_selection_frac, client_selection_strategy, other_info=None, subsets=0):
    np.random.seed(other_info)
    if client_selection_strategy == "random":
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))

    if client_selection_strategy == "subset":
        assert subsets!=0
        assert num_clients%subsets == 0
        per_num_clients = num_clients/subsets
        num_selected = max(int(client_selection_frac * per_num_clients), 1)
        tmp_arrys = np.split(np.arange(num_clients),subsets)
        selected_clients_set = None
        for i in range(subsets):  
            tmp_set = set(np.random.choice(tmp_arrys[i], num_selected, replace=False))
            if i==0:
                selected_clients_set = tmp_set
            else:
                selected_clients_set = selected_clients_set | tmp_set
        # print(selected_clients_set)

    return selected_clients_set
