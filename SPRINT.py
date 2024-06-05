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
                if sample[attr_index] in split_value:
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
        is_few_samples = False if self.min_samples is None else len(dataset) <= self.min_samples
        target_values = set(record[-1] for record in dataset if record[-1])
        grouped_dataset = [(value, [record for record in dataset if record[-1] == value]) for value in target_values]
        if len(grouped_dataset) == 1 or is_few_samples:
            sorted_grouped_dataset = deepcopy(grouped_dataset)
            sorted_grouped_dataset.sort(key=lambda group: len(group[1]))
            majority_class = sorted_grouped_dataset[0][0]
            return majority_class

        gini_values = []
        for attr_index, attr_type in enumerate(self.attribute_types):
            unique_values = list(set(record[attr_index] for record in dataset))
            unique_values.sort()
            if attr_type != 'string':
                split_points = unique_values[:-1]
            else:
                split_points = unique_values

            if len(split_points) == 0:
                continue
            for split_value in split_points:
                split_criteria = (attr_index, attr_type, split_value)

                left_subset, right_subset = [], []
                for record in dataset:
                    if attr_type == 'string':
                        if record[attr_index] in split_value:
                            left_subset.append(record)
                        else:
                            right_subset.append(record)
                    else:
                        if record[attr_index] <= split_value:
                            left_subset.append(record)
                        else:
                            right_subset.append(record)

                left_target_values = set(record[-1] for record in left_subset if record[-1])
                grouped_left = [(value, [record for record in left_subset if record[-1] == value]) for value in left_target_values]
                left_gini = 1
                for (class_label, group) in grouped_left:
                    proportion = len(group) / len(left_subset)
                    left_gini -= proportion ** 2

                right_target_values = set(record[-1] for record in right_subset if record[-1])
                grouped_right = [(value, [record for record in right_subset if record[-1] == value]) for value in right_target_values]
                right_gini = 1
                for (class_label, group) in grouped_right:
                    proportion = len(group) / len(right_subset)
                    right_gini -= proportion ** 2

                weighted_left_gini = len(left_subset) / len(dataset) * left_gini
                weighted_right_gini = len(right_subset) / len(dataset) * right_gini
                total_gini = weighted_left_gini + weighted_right_gini

                gini_values.append((split_criteria, total_gini))

        if len(gini_values) == 0:
            sorted_grouped_dataset = deepcopy(grouped_dataset)
            sorted_grouped_dataset.sort(key=lambda group: len(group[1]))
            majority_class = sorted_grouped_dataset[0][0]
            return majority_class

        best_split = min(gini_values, key=lambda item: item[1])
        
        left_subset, right_subset = [], []
        attr_index, attr_type, split_value = best_split[0]
        for record in dataset:
            if attr_type == 'string':
                if record[attr_index] in split_value:
                    left_subset.append(record)
                else:
                    right_subset.append(record)
            else:
                if record[attr_index] <= split_value:
                    left_subset.append(record)
                else:
                    right_subset.append(record)

        left_node = self.build_tree(left_subset)
        right_node = self.build_tree(right_subset)

        return (None, best_split[0], left_node, right_node)

    def __str__(self):
        return "SPRINT"
