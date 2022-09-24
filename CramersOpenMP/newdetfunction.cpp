#include <vector>
#include <cmath>
#include <iostream>
#include <mpi.h>

//code from https://codereview.stackexchange.com/questions/204135/determinant-using-gauss-elimination

double determinant(std::vector<std::vector<double>>& matrix) {
    int N = static_cast<int>(matrix.size());
    double det = 1;

    //MPI_Status status;
    //MPI_Init(NULL, NULL);
    ////Get process ID
    //int world_rank, world_size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    ////Get processes Number
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for (int i = 0; i < N; ++i) {

        double pivotElement = matrix[i][i];
        int pivotRow = i;
        for (int row = i + 1; row < N; ++row) {
            if (std::abs(matrix[row][i]) > std::abs(pivotElement)) {
                pivotElement = matrix[row][i];
                pivotRow = row;
            }
        }
        if (pivotElement == 0.0) {
            return 0.0;
        }
        if (pivotRow != i) {
            matrix[i].swap(matrix[pivotRow]);
            det *= -1.0;
        }
        det *= pivotElement;

        for (int row = i + 1; row < N; ++row) {
            for (int col = i + 1; col < N; ++col) {
                matrix[row][col] -= matrix[row][i] * matrix[i][col] / pivotElement;
            }
        }
    }
    /*MPI_Finalize();*/
    return det;
}





std::vector<double> solveCramer(std::vector<std::vector<double>>& equations) {
    //determinant
    

    int size = equations.size();

    std::vector<std::vector<double>> matrix(size);
    std::vector<double> column(size);
    for (int r = 0; r < size; ++r) {
        column[r] = equations[r][size];
        matrix[r].resize(size);
        for (int c = 0; c < size; ++c) {
            matrix[r][c] = equations[r][c];
            std::cout << matrix[r][c] << std::endl;
        }
    }


    
    

    //calculate Dx,y,z
    std::vector<double> answer(size);
    /*for (int i = 0; i < matrix.size(); ++i) {
        matrix.columnIndex(i);
        answer[i] = determinant(matrix)/ det;
    }*/
    return answer;
}

int main() {
    std::vector<std::vector<double>> equations = {
        { 2, -1,  5},
        { 3,  2,  2},
        { 1,  3,  3},

    };

    solveCramer(equations);
    /*std::cout << determinant(equations) << std::endl;*/
}