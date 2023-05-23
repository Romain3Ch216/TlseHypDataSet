import numpy as np
from ortools.sat.python import cp_model
from torch.utils.data import Subset
import datetime


__all__ = [
    'spatial_disjoint_split'
]


def spatial_disjoint_split(dataset, p_labeled, p_val, p_test, with_proportions=False, with_indices=False, fold : int = None, n_solutions: int = 1000, duplicate=False, timestamp=None):
    proportions, split = sat_split_solver(dataset, p_labeled, p_val, p_test, fold=fold, n_solutions=n_solutions, duplicate=duplicate, timestamp=timestamp)
    all_groups = np.unique(dataset.ground_truth['Group'])
    groups_in_labeled_set = all_groups[split == 0]
    groups_in_unlabeled_set = all_groups[split == 1]
    groups_in_validation_set = all_groups[split == 2]
    groups_in_test_set = all_groups[split == 3]

    def get_indices(groups_in_set, groups):
        indices = np.zeros_like(groups)
        for group in groups_in_set:
            indices += group == groups
        indices = np.where(indices == 1)[0]
        return indices

    indices = (get_indices(groups_in_labeled_set, dataset.samples[:, 1]),
               get_indices(groups_in_unlabeled_set, dataset.samples[:, 1]),
               get_indices(groups_in_validation_set, dataset.samples[:, 1]),
               get_indices(groups_in_test_set, dataset.samples[:, 1]))

    labeled_set = Subset(dataset, indices[0])
    unlabeled_set = Subset(dataset, indices[1])
    validation_set = Subset(dataset, indices[2])
    test_set = Subset(dataset, indices[3])

    if (with_proportions is False) and (with_indices is False):
        return labeled_set, unlabeled_set, validation_set
    elif with_proportions and with_indices:
        return labeled_set, unlabeled_set, validation_set, test_set, proportions, indices
    elif with_proportions:
        return labeled_set, unlabeled_set, validation_set, test_set, proportions
    elif with_indices:
        return labeled_set, unlabeled_set, validation_set, test_set, indices


def sat_split_solver(dataset, p_labeled: float, p_val: float, p_test: float, n_solutions: int = 1000, fold: int = None, duplicate = False, timestamp=None) -> np.ndarray:
    """
    Solves a SAT problem to optimally split the ground truth in a labeled, unlabeled and test sets.

    Constraints are defined as follows:
        - each group should belong to one and only one set,
        - each class occupies at least p_labeled * total_class_area m^2 in the labeled set,
        - each class occupies at least p_test * total_class_area m^2 in the test set,
    The objective is to maximize the area of the unlabeled set.

    In other words, the labeled and test sets mandatory contain each class but the unlabeled set may lack some.

    :param areas: array of size (n_groups x n_classes) which contains the area (in m^2) that each class occupies
    in every spatial groups (e.g. a group of ground truth polygons)
    :param p_labeled: minimum proportion of area to put in the labeled split
    :param p_test: minimum proportion of area to put in the test split
    :return: array of size (1 x n_groups) which values are in {0, 1, 2} (0: labeled set, 1: unlabeled set, 2: test set)
    """
    n_sets = 4
    areas = dataset.areas
    total_area = int(np.sum(areas))

    if duplicate is False and dataset.split_already_computed(p_labeled, p_val, p_test, timestamp):
        solutions = dataset.load_splits(p_labeled, p_val, p_test, timestamp)
    else:
        # Compute minimum areas for each class
        non_zeros_groups = np.sum(areas > 0, axis=0)
        prop = np.array([p_labeled, p_val, p_test])
        prop = np.sort(prop)
        total_l_area, total_v_area, total_t_area = [], [], []
        for class_id in range(areas.shape[1]):
            if non_zeros_groups[class_id] <= 5:
                class_areas = areas[:, class_id]
                class_areas = class_areas[class_areas > 0]
                class_areas = np.sort(class_areas)
                if p_labeled > 0:
                    total_l_area.append(class_areas[np.where(prop == p_labeled)[0][0]] / p_labeled)
                else:
                    total_l_area.append(0)
                if p_val > 0:
                    total_v_area.append(class_areas[np.where(prop == p_val)[0][0]] / p_val)
                else:
                    total_v_area.append(0)
                if p_test > 0:
                    total_t_area.append(class_areas[np.where(prop == p_test)[0][0]] / p_test)
                else:
                    total_t_area.append(0)
            else:
                total_v_area.append(np.sum(areas[:, class_id]))
                total_l_area.append(np.sum(areas[:, class_id]))
                total_t_area.append(np.sum(areas[:, class_id]))
        total_l_area, total_v_area, total_t_area = assert_feasible(total_l_area, total_v_area, total_t_area, p_labeled, p_val, p_test, areas)
        # Initialize SAT model
        model = cp_model.CpModel()
        # sets is a dict which keys (i, j) are linked to values equal to 1 if group i is in set j, 0 otherwise
        sets = {}
        for group_id in range(areas.shape[0]):
            for set_ in range(n_sets):
                sets[group_id, set_] = model.NewBoolVar(name='group_%i_set_%i' % (group_id, set_))
            # Each group belongs to one and only one set
            model.Add(sum([sets[group_id, k] for k in range(n_sets)]) == 1) #, "group_in_set")

        for class_id in range(areas.shape[1]):
            # Minimum area constraint in the labeled set
            model.Add(
                sum([areas[group_id, class_id] * sets[group_id, 0]
                     for group_id in range(areas.shape[0])]) >= int(p_labeled * total_l_area[class_id]))
            # Minimum area constraint in the validation set
            model.Add(
                sum([areas[group_id, class_id] * sets[group_id, 2]
                     for group_id in range(areas.shape[0])]) >= int(p_val * total_v_area[class_id]))
            # Minimum area constraint in the test set
            model.Add(
                sum([areas[group_id, class_id] * sets[group_id, 3]
                     for group_id in range(areas.shape[0])]) >= int(p_test * total_t_area[class_id]))

        # Upper bounds of labeled and test areas
        labeled_area = model.NewIntVar(0, total_area, name="labeled_area")
        val_area = model.NewIntVar(0, total_area, name="val_area")
        test_area = model.NewIntVar(0, total_area, name="test_area")
        model.Add(
            labeled_area >= sum([
                sum([
                    areas[group_id, class_id] * sets[group_id, 0]
                    for group_id in range(areas.shape[0])])
                for class_id in range(areas.shape[1])]))
        model.Add(
            val_area >= sum([
                sum([
                    areas[group_id, class_id] * sets[group_id, 2]
                    for group_id in range(areas.shape[0])])
                for class_id in range(areas.shape[1])]))
        model.Add(
            test_area >= sum([
                sum([
                    areas[group_id, class_id] * sets[group_id, 3]
                    for group_id in range(areas.shape[0])])
                for class_id in range(areas.shape[1])]))

        # Objective function
        model.Minimize(labeled_area + val_area + test_area)
        solver = cp_model.CpSolver()
        solution_printer = VarArraySolutionPrinterWithLimit(sets, n_solutions)
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(model, solution_printer)
        # print('Status = %s' % solver.StatusName(status))
        print('Number of solutions found: %i' % solution_printer.solution_count())
        # assert solution_printer.solution_count() == 100
        solutions = solution_printer.solutions()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset.save_splits(solutions, p_labeled, p_val, p_test, timestamp)

    if fold is None:
        random_fold = np.random.randint(3*len(solutions)//4, len(solutions), size=1)
        fold = random_fold[0]
    else:
        assert fold < len(solutions), "Fold must be inferior to the number of solutions, i.e. {}".format(len(solutions))
    solution = solutions[fold]
    array_solution = np.zeros((n_sets, areas.shape[0]))
    for k, v in solution.items():
        array_solution[k[1], k[0]] = v
    array_solution = np.argmax(array_solution, axis=0)

    labeled_areas = np.sum(areas[array_solution == 0, :], axis=0) / np.sum(areas, axis=0)
    unlabeled_areas = np.sum(areas[array_solution == 1, :], axis=0) / np.sum(areas, axis=0)
    validation_areas = np.sum(areas[array_solution == 2, :], axis=0) / np.sum(areas, axis=0)
    test_areas = np.sum(areas[array_solution == 3, :], axis=0) / np.sum(areas, axis=0)
    proportions = [labeled_areas, unlabeled_areas, validation_areas, test_areas]
    return proportions, array_solution


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Save intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit
        self.__solutions = {}

    def on_solution_callback(self):
        self.__solution_count += 1
        print('Solution %i' % self.__solution_count)
        self.__solutions[self.__solution_count] = {}
        for v, var in self.__variables.items():
            self.__solutions[self.__solution_count][v] = self.Value(var)
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count

    def solutions(self):
        return self.__solutions


def assert_feasible(total_l_area,
                    total_v_area,
                    total_t_area,
                    p_labeled,
                    p_val,
                    p_test,
                    areas):
    feasibility = True
    n_classes = areas.shape[1]
    for class_id in range(n_classes):
        total_areas = np.array([
            total_l_area[class_id] * p_labeled,
            total_v_area[class_id] * p_val,
            total_t_area[class_id] * p_test]).astype(int)
        prop = np.sort([p_labeled, p_val, p_test])
        total_areas = np.sort(total_areas)
        set_order = np.argsort(total_areas)
        area = areas[:, class_id]
        area = np.sort(area)
        current_area = 0
        i = 0
        j = 0
        while i < 3 and j < len(area):
            if current_area < total_areas[i]:
                current_area += area[j]
                j+= 1
            elif i == 2:
                j+=1
            else:
                i+= 1
                current_area = 0
        feasible = (i >= 2) and (current_area >= total_areas[i])
        if feasible is False:
            n_groups = []
            cum_area = np.cumsum(area[area>0])
            i = int(prop[0] * len(cum_area))
            for k, set_id in enumerate(set_order):
                n_groups.append(cum_area[i])
                if k < 2:
                    i += int(prop[k+1] * len(cum_area))
            total_l_area[class_id] = n_groups[np.where(set_order == 0)[0][0]]
            total_v_area[class_id] = n_groups[np.where(set_order == 1)[0][0]]
            total_t_area[class_id] = n_groups[np.where(set_order == 2)[0][0]]
            print(f'Class {class_id+1}: unresolved constraints')
        feasibility = feasibility and feasible
    return total_l_area, total_v_area, total_t_area

