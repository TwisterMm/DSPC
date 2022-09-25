#include <vector>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <omp.h>
#include<iterator>

using Matrix = std::vector<std::vector<double>>;
int rank, size;
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    auto it = v.cbegin();
    auto end = v.cend();

    os << '[';
    if (it != end) {
        os << *it;
        it = std::next(it);
    }
    while (it != end) {
        os << ", " << *it;
        it = std::next(it);
    }
    return os << ']';
}



Matrix squareMatrix(size_t n) {
    Matrix m;
    for (size_t i = 0; i < n; i++) {
        std::vector<double> inner;
        for (size_t j = 0; j < n; j++) {
            inner.push_back(nan(""));
        }
        m.push_back(inner);
    }
    return m;
}

Matrix minor(const Matrix& a, int x, int y) {
    auto length = a.size() - 1;
    auto result = squareMatrix(length);
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            if (i < x && j < y) {
                result[i][j] = a[i][j];
            }
            else if (i >= x && j < y) {
                result[i][j] = a[i + 1][j];
            }
            else if (i < x && j >= y) {
                result[i][j] = a[i][j + 1];
            }
            else {
                result[i][j] = a[i + 1][j + 1];
            }
        }
    }
    return result;
}

double det(const Matrix& a) {
    if (a.size() == 1) {
        return a[0][0];
    }

    int sign = 1;
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += sign * a[0][i] * det(minor(a, 0, i));
        sign *= -1;
    }
    return sum;
}




std::vector<double> solveCramer(Matrix& equations) {
    //get matrix 
    //get determinant
    //calculate answer

    int size = equations.size();

    if (std::any_of(equations.cbegin(), equations.cend(),
        [size](const std::vector<double>& a) { return a.size() != size + 1; }
    )) {
        throw std::runtime_error("Each equation must have the expected size.");
    }



    Matrix matrix(size);
    std::vector<double> column(size);
    

    for (int r = 0; r < size; ++r) {
        //if (rank == 0) {
        //    column[r] = equations[r][size];//get the answer column
        //    matrix[r].resize(size);//resize each row
        //    MPI_Scatter(&equations, 1, MPI_DOUBLE, &column, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //}
        column[r] = equations[r][size];//get the answer column
        matrix[r].resize(size);//resize each row
        for (int c = 0; c < size; ++c) {
            matrix[r][c] = equations[r][c];//assign each element to the matrix            
        }
    }

    double determinant = det(matrix);
    //prepare special matrix
    std::vector<double> determinantArray(size);

    
        
    for (int col = 0; col < size; col++)
    {
        if (rank == 0) {
            std::vector<std::vector<double>> detX;
            std::copy(matrix.begin(), matrix.end(), back_inserter(detX));
            MPI_Scatter(&determinantArray, 1, MPI_DOUBLE, &detX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        for (int row = 0; row < size; row++) {
            detX[row][col] = column[row];
        }
        }
        determinantArray[col] = det(detX);
        MPI_Allgather(&determinantArray, 1, MPI_DOUBLE, &determinantArray, 4, MPI_DOUBLE, MPI_COMM_WORLD);
    }
        
    

    std::vector<double> answer(matrix.size());
    for (int i = 0; i < size; ++i) {        
        answer[i] = determinantArray[i] / determinant;
    }    
    return answer;
}

int main(int argc, char* argv[]) {
    double start_time, end_time;
    std::vector<std::vector<double>> equations = {
        { 2, -1,  5,  1, -3},
        { 3,  2,  2, -6, -32},
        { 1,  3,  3, -1, -47},
        { 5, -2, -3,  3, 49},
    };

    MPI_Init(&argc, &argv);
    //Get processes Number
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    //Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time_MPI, end_time_MPI;
    start_time_MPI = MPI_Wtime();
    auto solution = solveCramer(equations);
    end_time_MPI = MPI_Wtime();
    MPI_Finalize();

    std::cout << "time taken in seconds: " << end_time_MPI << "s\n";
    std::cout << solution << '\n';
    return 0;

}