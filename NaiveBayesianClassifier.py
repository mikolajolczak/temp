import random

class NaiveBayesianClassifier:
    def __init__(self, smoothing=0):
        self.class_histograms = None
        self.dataset_size = None
        self.attribute_count = None
        self.smoothing_factor = smoothing
        self.class_distribution = dict()

    def fit(self, dataset, column_info):
        self.attribute_count = len(dataset[0]) - 1
        self.class_histograms = dict()
        self.class_distribution = dict()

        for record in dataset:
            cls = record[-1]

            if cls not in self.class_histograms:
                self.class_histograms[cls] = [{} for _ in range(self.attribute_count)]

            for i in range(self.attribute_count):
                attr_value = record[i]
                if attr_value not in self.class_histograms[cls][i]:
                    self.class_histograms[cls][i][attr_value] = 1
                else:
                    self.class_histograms[cls][i][attr_value] += 1

            if cls not in self.class_distribution:
                self.class_distribution[cls] = 1
            else:
                self.class_distribution[cls] += 1

        self.dataset_size = len(dataset)

    def predict(self, instance):
        class_probabilities = []

        for cls in self.class_histograms:
            probability = 1
            for attr_idx in range(len(instance)):
                if self.smoothing_factor:
                    if instance[attr_idx] in self.class_histograms[cls][attr_idx]:
                        probability *= self.class_histograms[cls][attr_idx][instance[attr_idx]] + self.smoothing_factor
                    else:
                        probability *= self.smoothing_factor

                    probability /= self.class_distribution[cls] + self.attribute_count * self.smoothing_factor
                else:
                    if instance[attr_idx] in self.class_histograms[cls][attr_idx]:
                        probability *= self.class_histograms[cls][attr_idx][instance[attr_idx]] / self.class_distribution[cls]
                    else:
                        probability = 0
                        break

            probability *= self.class_distribution[cls] / self.dataset_size
            class_probabilities.append((cls, probability))

        class_probabilities.sort(key=lambda x: x[1], reverse=True)

        max_probability = class_probabilities[0][1]
        tied_classes = [cls_prob[0] for cls_prob in class_probabilities if cls_prob[1] == max_probability]

        if len(tied_classes) > 1:
            return random.choice(tied_classes)

        return class_probabilities[0][0]

    def __str__(self):
        return f"Naive Bayesian Classifier with {self.smoothing_factor} smoothing"
