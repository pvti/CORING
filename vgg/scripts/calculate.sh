python -m calculation.calculate_svd --decomposer hosvd &&

python -m calculation.calculate_correlation_svd --correlation cosine_sim Euclide_dis VBD_dis &&

python -m calculation.calculate_correlation --correlation cosine_sim Euclide_dis VBD_dis &&

python -m calculation.merge_correlation &&

python -m calculation.calculate_rank