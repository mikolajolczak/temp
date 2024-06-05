import numpy as np
from itertools import combinations
from collections import defaultdict

class LazyClassificationwithContrastPatterns:
    def __init__(self):
        self.obfuscated_data = None
        self.obfuscation_threshold = 0.01
        self.class_freq = {}
        self.feature_metadata = None
        self.default_label = None
        self.binary_values = None
        self.categorical_columns = []
        self.categorical_values = defaultdict(set)

    def fit(self, data, metadata):
        self.obfuscated_data = np.array(data, dtype=object)
        self.feature_metadata = metadata
        self.binary_values = [2 ** i for i in range(self.obfuscated_data.shape[1] - 1)]
        
        for idx, info in enumerate(metadata[1]):
            if info == 'string':
                self.categorical_columns.append(idx)
                self.categorical_values[idx] = set(self.obfuscated_data[:, idx])

        for entry in self.obfuscated_data:
            label = entry[-1]
            if label not in self.class_freq:
                self.class_freq[label] = 0
            self.class_freq[label] += 1

        self.default_label = max(self.class_freq, key=self.class_freq.get)

    def predict(self, instance):
        pattern_matches = defaultdict(lambda: defaultdict(int))
        for idx, entry in enumerate(self.obfuscated_data):
            bit_mask = 0
            for feat_idx, feat_val in enumerate(instance):
                if feat_idx in self.categorical_columns:
                    if feat_val in self.categorical_values[feat_idx]:
                        bit_mask += self.binary_values[feat_idx]
                else:
                    if self.feature_metadata and self.feature_metadata[0] and self.feature_metadata[1]:
                        if 0 <= feat_idx < len(self.feature_metadata[0]) and 0 <= feat_idx < len(self.feature_metadata[1]):
                            margin = self.obfuscation_threshold * self.feature_metadata[0][feat_idx].get('range', 0) if self.feature_metadata[1][feat_idx] == 'int' else 0
                            if float(self.obfuscated_data[idx][feat_idx]) - margin <= instance[feat_idx] <= float(self.obfuscated_data[idx][feat_idx]) + margin:
                                bit_mask += self.binary_values[feat_idx]

            if bit_mask > 0:
                label = entry[-1]
                pattern_matches[bit_mask][label] += 1

        score = {}

        for length in range(1, len(instance) + 1):
            for comb in combinations(range(len(instance)), length):
                bit_sum = sum(self.binary_values[pos] for pos in comb)
                matched_keys = [key for key in pattern_matches if bit_sum & key == bit_sum]
                if matched_keys:
                    detected_class = None
                    count = 0
                    consistent = True
                    for key in matched_keys:
                        if len(pattern_matches[key]) > 1:
                            consistent = False
                            break
                        label = next(iter(pattern_matches[key]))
                        count = pattern_matches[key][label]
                        if detected_class is None or detected_class == label:
                            count += count
                            detected_class = label
                        else:
                            consistent = False
                            break
                    if consistent:
                        if detected_class not in score:
                            score[detected_class] = 0
                        score[detected_class] = max(score[detected_class], count)

            if len(score) == len(self.class_freq):
                break

        if score:
            sorted_scores = sorted(score.items(), key=lambda item: item[1] / self.class_freq[item[0]], reverse=True)
            return sorted_scores[0][0]
        return self.default_label

    def __str__(self):
        return "LazyClassificationwithContrastPatterns"
