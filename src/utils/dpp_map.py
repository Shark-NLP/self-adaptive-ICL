import numpy as np
import math
import numpy


def fast_map_dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def greedy_map_dpp(kernel_matrix, max_length):
    """
    greedy map
    reference: http://jgillenw.com/dpp-map.html
    paper: Near-Optimal MAP Inference for Determinantal Point Processes
    """
    selected_items = []
    item_size = kernel_matrix.shape[0]
    U = list(range(0, item_size))
    num_left = item_size

    while len(U) > 0:
        scores = np.diag(kernel_matrix)
        # Select the max-scoring addition to the chosen set.
        max_loc = np.argmax(scores)
        max_score = scores[max_loc]

        if max_score < 1 or len(selected_items) == max_length:
            break
        selected_items.append(U[max_loc])
        del U[max_loc]

        # Compute the new kernel, conditioning on the current selection.
        inc_ids = list(range(0, max_loc)) + list(range(max_loc + 1, num_left))

        kernel_matrix = numpy.linalg.inv(
            kernel_matrix + np.diag([1] * (max_loc) + [0] + [1] * (num_left - max_loc - 1)))
        num_left -= 1
        kernel_matrix = numpy.linalg.inv(kernel_matrix[np.ix_(inc_ids, inc_ids)]) - np.eye(num_left)

    return selected_items


if __name__ == "__main__":
    import time

    item_size = 100
    feature_dimension = 1000
    max_length = 50

    scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)
    feature_vectors = np.random.randn(item_size, feature_dimension)

    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T)
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))

    t = time.time()
    result = fast_map_dpp(kernel_matrix, max_length)
    print(result)
    print('fast dpp algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))

    t = time.time()
    result = greedy_map_dpp(kernel_matrix, max_length)
    print(result)
    print('greedy dpp algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
