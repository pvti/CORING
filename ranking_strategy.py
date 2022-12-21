import numpy as np


def compare_min(row, col, matrix, dis=1):
    """Compare minimum distance/similarity from row, col to other elements
    Matrix must be symmetrix
    """
    if dis == 1:
        min_row = np.min(matrix[row, :])
        min_col = np.min(matrix[:, col])

        return col if min_row > min_col else row

    else:
        max_row = np.max(matrix[row, :])
        max_col = np.max(matrix[:, col])

        return col if max_row < max_col else row


def compare_sum(row, col, matrix, inf, dis=1):
    """Compare sum of distance/similarity from row, col to other elements
    Matrix must be symmetrix
    """
    num_row = matrix.shape[0]
    if dis == 1:
        min_row = min_col = 0
        for i in range(num_row):
            if matrix[row, i] != inf:
                min_row += matrix[row, i]
            if matrix[i, col] != inf:
                min_col += matrix[i, col]

        return col if min_row > min_col else row

    else:
        max_row = max_col = 0
        for i in range(num_row):
            if matrix[row, i] != inf:
                max_row += matrix[row, i]
            if matrix[i, col] != inf:
                max_col += matrix[i, col]

        return col if max_row < max_col else row


def optimal_arg(x, optimal=1):
    """Return argmax/argmin based on optimal value.
        Default: optimal = 1 => max
    """
    if optimal == 1:
        return x.argmin()

    return x.argmax()


def get_saliency(mat, strategy="sum", dis=1):
    """Sort saliency based on distance/similarity matrix
    """
    num_row = mat.shape[0]
    saliency = np.full(num_row, num_row-1, dtype=np.float32)

    if strategy == "sum":
        for i in range(num_row):
            saliency[i] = dis * mat[i, :].sum()

    else:
        inf = dis*float('inf')
        for i in range(num_row):
            mat[i, i] = inf
        for i in range(num_row-1):
            row, col = np.unravel_index(optimal_arg(mat, dis), mat.shape)
            mat[row, col] = inf
            mat[col, row] = inf
            # default strategy == "min_sum"
            index = compare_sum(row, col, mat, inf, dis)
            if strategy == "min_min":
                index = compare_min(row, col, mat, dis)
            mat[index, :] = inf
            mat[:, index] = inf
            saliency[index] = i

    return saliency
