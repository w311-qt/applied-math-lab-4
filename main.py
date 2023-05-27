import seidel, matrix_generators, lu_decomposition
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


print("Dominance matrix:")

lu_time = []
seidel_time = []
size = []
for i in range(10, 200, 10):
    A = matrix_generators.generate_diagonal_dominance_matrix(i)
    b = matrix_generators.generate_vector(A)
    solution1_1, time1 = lu_decomposition.system_solution(A, b)
    solution1_2, time2 = seidel.seidel_method(A, b)
    size.append(i)
    lu_time.append(time1)
    seidel_time.append(time2)
    print("K = ", i, "\nLu time: ", time1, "\nSeidel time: ", time2)

plt.title("Dependency of execution time from matrix size in LU system solution and Seidel iterative method")
plt.xlabel("size")
plt.ylabel("time")
plt.grid()
plt.plot(size, lu_time, size, seidel_time)
plt.show()