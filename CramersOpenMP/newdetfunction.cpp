#include <vector>
#include <cmath>
#include <iostream>

//code from https://codereview.stackexchange.com/questions/204135/determinant-using-gauss-elimination

double determinant(std::vector<std::vector<double>>& matrix) {
    int N = static_cast<int>(matrix.size());
    double det = 1;

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

    return det;
}


int main() {
    std::vector<std::vector<double>> equations = {
        { 2, -1,  5},
        { 3,  2,  2},
        { 1,  3,  3},

    };
    std::cout << determinant(equations) << std::endl;
}