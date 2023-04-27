import numpy as np
from ortools.sat.python import cp_model
from torch.utils.data import Subset


__all__ = [
    'spatial_disjoint_split'
]


def spatial_disjoint_split(dataset, p_labeled, p_val, p_test):
    proportions, split = sat_split_solver(dataset, p_labeled, p_val, p_test)
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

    labeled_set = Subset(dataset, get_indices(groups_in_labeled_set, dataset.samples[:, 1]))
    unlabeled_set = Subset(dataset, get_indices(groups_in_unlabeled_set, dataset.samples[:, 1]))
    validation_set = Subset(dataset, get_indices(groups_in_validation_set, dataset.samples[:, 1]))
    test_set = Subset(dataset, get_indices(groups_in_test_set, dataset.samples[:, 1]))
    return labeled_set, unlabeled_set, validation_set, test_set, proportions


def sat_split_solver(dataset, p_labeled: float, p_val: float, p_test: float, n_solutions: int = 1000) -> np.ndarray:
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

    if dataset.split_already_computed(p_labeled, p_val, p_test):
        solutions = dataset.load_splits(p_labeled, p_val, p_test)
    else:
        # Compute minimum areas for each class
        non_zeros_groups = np.sum(areas > 0, axis=0)
        total_l_area, total_v_area, total_t_area = [], [], []
        for class_id in range(areas.shape[1]):
            total_v_area.append(np.sum(areas[:, class_id]))
            total_l_area.append(np.sum(areas[:, class_id]))
            total_t_area.append(np.sum(areas[:, class_id]))
        import pdb; pdb.set_trace()
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

        dataset.save_splits(solutions, p_labeled, p_val, p_test)
#
    random_fold = np.random.randint(3*len(solutions)//4, len(solutions), size=1)
    random_fold = random_fold[0]
    solution = solutions[random_fold]
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
