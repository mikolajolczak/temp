import random
from math import floor

FLOAT = 'float'
INTEGER = 'int'
STRING = 'string'

class DatasetParser:
    def __init__(self, input_data, split_ratio, column_types, num_buckets):
        self.raw_data, self.metadata = [], []
        self.column_types = column_types
        self.num_buckets = num_buckets
        self.split_ratio = split_ratio

        for _ in column_types:
            self.metadata.append({'min': None, 'max': None})

        for record in input_data:
            if len(record) <= 1:
                continue

            for idx in range(min(len(record) - 1, len(column_types))):
                if column_types[idx] == STRING:
                    if record[idx] not in self.metadata[idx]:
                        self.metadata[idx][record[idx]] = len(self.metadata[idx])
                    record[idx] = self.metadata[idx][record[idx]]
                else:
                    value = float(record[idx])
                    if self.metadata[idx]['min'] is None or value < self.metadata[idx]['min']:
                        self.metadata[idx]['min'] = value
                    if self.metadata[idx]['max'] is None or value > self.metadata[idx]['max']:
                        self.metadata[idx]['max'] = value
                    record[idx] = value

            self.raw_data.append(record)

        for idx in range(len(self.metadata)):
            if self.column_types[idx] == FLOAT:
                if self.metadata[idx]['min'] is not None and self.metadata[idx]['max'] is not None:
                    self.metadata[idx]['range'] = self.metadata[idx]['max'] - self.metadata[idx]['min']

        random.shuffle(self.raw_data)
        self.total_data_points = len(self.raw_data)

    def bin_record(self, record):
        binned_record = []
        for idx in range(len(record) - 1):
            if idx < len(self.column_types) and self.column_types[idx] != STRING:
                min_val = self.metadata[idx]['min']
                max_val = self.metadata[idx]['max']
                if min_val is not None and max_val is not None:
                    bin_index = floor(self.num_buckets * (record[idx] - min_val) / (max_val - min_val))
                    if bin_index >= self.num_buckets:
                        bin_index = self.num_buckets - 1
                    binned_record.append(bin_index)
                else:
                    binned_record.append(0)
            else:
                binned_record.append(record[idx])
        binned_record.append(record[-1])
        return binned_record

    def get_training_set(self, binned=False):
        split_idx = floor(self.total_data_points * self.split_ratio)
        if binned:
            return [self.bin_record(record) for record in self.raw_data[:split_idx]]
        else:
            return self.raw_data[:split_idx]

    def get_test_set(self, binned=False):
        split_idx = floor(self.total_data_points * self.split_ratio)
        if binned:
            return [self.bin_record(record) for record in self.raw_data[split_idx:]]
        else:
            return self.raw_data[split_idx:]

    def get_metadata(self):
        return self.metadata, self.column_types
