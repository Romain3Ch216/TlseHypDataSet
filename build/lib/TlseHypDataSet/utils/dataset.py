import numpy as np
from ortools.sat.python import cp_model
from torch.utils.data import Subset
import pickle as pkl


__all__ = [
    'DisjointDataSplit',
    'sat_split_solver'
]


class DisjointDataSplit:
    """
    A class to produce spatially disjoint train / test splits of the ground truth as described in ...
    """
    def __init__(self, dataset, split=None, proportions=None, file=None, n_solutions=1000):
        """
        :param dataset: A TlseHypDataSet object
        :param split: An array of size (1 x n_groups) specifying the assignment of each group to a set
        :param proportions: A list in the following format: [p_labeled, p_val and p_test].\
         If the argument split is not given, compute a split such that the proportions of pixels in\
         the labeled training set, the validation set and the test set are greater than p_labeled, p_val and p_test,\
          respectively.
        :param file: Path to a file where a split is saved in a pickle format
        :param n_solutions: the maximum number of solutions for the SAT solver (used only with the proportions argument)
        """
        self.dataset = dataset
        self.areas = self.dataset.areas
        if split is None and file is None:
            self.split_ = sat_split_solver(dataset,
                                           p_labeled=proportions[0],
                                           p_val=proportions[1],
                                           p_test=proportions[2],
                                           n_solutions=n_solutions)

        elif split is None and file is not None:
            with open(file, 'rb') as f:
                self.split_ = pkl.load(f)
        else:
            self.split_ = split

    @property
    def groups_(self):
        """
        :return: A dict whose keys are sets and values are lists with assigned groups
        """
        all_groups = np.unique(self.dataset.ground_truth['Group'])
        all_groups = all_groups[np.isnan(all_groups) == False]
        groups = {
            'labeled': all_groups[self.split_ == 0],
            'unlabeled': all_groups[self.split_ == 1],
            'validation': all_groups[self.split_ == 2],
            'test': all_groups[self.split_ == 3]
        }
        return groups

    @property
    def sets_(self):
        """
        :return: A dict whose keys are sets and values are labeled training, unlabeled training, validation and test\
        Pytorch datasets
        """
        indices = self.indices_
        sets = {
            'train': Subset(self.dataset, indices['train']),
            'labeled_pool': Subset(self.dataset, indices['labeled_pool']),
            'unlabeled_pool': Subset(self.dataset, indices['unlabeled_pool']),
            'validation': Subset(self.dataset, indices['validation']),
            'test': Subset(self.dataset, indices['test'])
        }
        return sets

    @property
    def indices_(self):
        """
        :return: A dict whose keys are sets and values are sample indices in the TlseHypDataSet
        """
        def get_indices(groups_in_set, groups):
            indices = np.zeros_like(groups)
            for group in groups_in_set:
                indices += group == groups
            indices = np.where(indices == 1)[0]
            return indices

        def get_unlabeled_indices(groups):
            indices = np.zeros_like(groups)
            indices += groups == -1
            indices = np.where(indices == 1)[0]
            return indices

        groups = self.groups_
        indices = {
            'train': get_indices(groups['labeled'], self.dataset.samples[:, 1]),
            'labeled_pool': get_indices(groups['unlabeled'], self.dataset.samples[:, 1]),
            'unlabeled_pool': get_unlabeled_indices(self.dataset.samples[:, 1]),
            'validation': get_indices(groups['validation'], self.dataset.samples[:, 1]),
            'test': get_indices(groups['test'], self.dataset.samples[:, 1]),

        }
        return indices

    @property
    def proportions_(self):
        labeled_areas = np.sum(self.areas[self.split_ == 0, :], axis=0) / np.sum(self.areas, axis=0)
        unlabeled_areas = np.sum(self.areas[self.split_ == 1, :], axis=0) / np.sum(self.areas, axis=0)
        validation_areas = np.sum(self.areas[self.split_ == 2, :], axis=0) / np.sum(self.areas, axis=0)
        test_areas = np.sum(self.areas[self.split_ == 3, :], axis=0) / np.sum(self.areas, axis=0)
        proportions = {
            'train': labeled_areas,
            'labeled_pool': unlabeled_areas,
            'validation': validation_areas,
            'test': test_areas
        }
        return proportions


def sat_split_solver(dataset,
                     p_labeled: float,
                     p_val: float,
                     p_test: float,
                     n_solutions: int = 1000) -> np.ndarray:
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

    # Compute minimum areas for each class
    prop = np.array([p_labeled, p_val, p_test])
    total_l_area, total_v_area, total_t_area = [], [], []
    for class_id in range(areas.shape[1]):
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
    # status = solver.Solve(model, solution_printer)
    # print('Status = %s' % solver.StatusName(status))
    print('Number of solutions found: %i' % solution_printer.solution_count())
    # assert solution_printer.solution_count() == n_solutions
    solutions = solution_printer.solutions()
    last_solution = list(solutions.keys())[-1]
    final_solution = solutions[last_solution]
    array_solution = np.zeros((n_sets, areas.shape[0]))
    for k, v in final_solution.items():
        array_solution[k[1], k[0]] = v
    array_solution = np.argmax(array_solution, axis=0)
    return array_solution


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
    """
    Assert is the problem is feasible, else decrease total_l_area, total_v_area, total_t_area
    :param total_l_area: 1 x n_classes array with minimum pixels required by classe in the train set
    :param total_v_area: 1 x n_classes array with minimum pixels required by class in the validation set
    :param total_t_area: 1 x n_classes array with minimum pixels required by class in the train set
    :param p_labeled: average proportion of pixels in the train set
    :param p_val: average proportion of pixels in the validation set
    :param p_test: average proportion of pixels in the test set
    :param areas: n_groups x n_classes array with number of pixels by group and by class
    """
    feasibility = True
    n_classes = areas.shape[1]
    for class_id in range(n_classes):
        total_areas = np.array([
            total_l_area[class_id] * p_labeled,
            total_v_area[class_id] * p_val,
            total_t_area[class_id] * p_test]).astype(int)
        prop = np.sort([p_labeled, p_val, p_test])
        total_areas, set_order = np.sort(total_areas), np.argsort(total_areas)
        # set_order = np.argsort(total_areas)
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
            cum_area = np.cumsum(area[area > 0])
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

