from copy import deepcopy

class SPRINT:
    def __init__(self, min_samples=None):
        self.attribute_types = None
        self.min_samples = min_samples
        self.decision_tree = None

    def fit(self, dataset, attribute_info):
        self.attribute_types = attribute_info
        self.decision_tree = self.build_tree(dataset)

    def predict(self, sample):
        node = self.decision_tree
        while isinstance(node, tuple):
            attr_index, attr_type, split_value = node[1]
            if attr_type == 'string':
                if sample[attr_index] == split_value:
                    node = node[2]
                else:
                    node = node[3]
            else:
                if sample[attr_index] <= split_value:
                    node = node[2]
                else:
                    node = node[3]
        return node

    def build_tree(self, dataset):
        if self.min_samples is not None and len(dataset) <= self.min_samples:
            return self.majority_class(dataset)

        target_values = set(record[-1] for record in dataset)
        if len(target_values) == 1:
            return next(iter(target_values))

        best_split = self.find_best_split(dataset)
        if best_split is None:
            return self.majority_class(dataset)

        left_subset, right_subset = self.split_dataset(dataset, best_split)
        left_node = self.build_tree(left_subset)
        right_node = self.build_tree(right_subset)

        return (None, best_split, left_node, right_node)

    def majority_class(self, dataset):
        target_values = [record[-1] for record in dataset]
        return max(set(target_values), key=target_values.count)

    def find_best_split(self, dataset):
        gini_values = []
        for attr_index, attr_type in enumerate(self.attribute_types):
            unique_values = sorted(set(record[attr_index] for record in dataset))
            if attr_type == 'string':
                split_points = unique_values
            else:
                split_points = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for split_value in split_points:
                split_criteria = (attr_index, attr_type, split_value)
                left_subset, right_subset = self.split_dataset(dataset, split_criteria)
                if not left_subset or not right_subset:
                    continue

                gini = self.calculate_gini(left_subset, right_subset)
                gini_values.append((split_criteria, gini))

        if not gini_values:
            return None

        best_split = min(gini_values, key=lambda item: item[1])
        return best_split[0]

    def split_dataset(self, dataset, split_criteria):
        attr_index, attr_type, split_value = split_criteria
        left_subset = []
        right_subset = []
        for record in dataset:
            if attr_type == 'string':
                if record[attr_index] == split_value:
                    left_subset.append(record)
                else:
                    right_subset.append(record)
            else:
                if record[attr_index] <= split_value:
                    left_subset.append(record)
                else:
                    right_subset.append(record)
        return left_subset, right_subset

    def calculate_gini(self, left_subset, right_subset):
        total_size = len(left_subset) + len(right_subset)
        weighted_gini = 0
        for subset in [left_subset, right_subset]:
            size = len(subset)
            if size == 0:
                continue
            class_counts = [record[-1] for record in subset]
            score = sum((class_counts.count(class_val) / size) ** 2 for class_val in set(class_counts))
            weighted_gini += (1 - score) * (size / total_size)
        return weighted_gini

    def __str__(self):
        return "SPRINT"
