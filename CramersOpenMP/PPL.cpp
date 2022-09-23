#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>
#include <ppl.h>
#include <concurrent_vector.h> //include concurrent vector
#include <tuple> //include tuple

using namespace std;
using namespace concurrency;
class SubMatrix {
    const std::vector<std::vector<double>>* source;
    std::vector<double> replaceColumn;
    const SubMatrix* prev;
    size_t sz;
    int colIndex = -1;

public:
    SubMatrix(const std::vector<std::vector<double>>& src, const std::vector<double>& rc) : source(&src), replaceColumn(rc), prev(nullptr), colIndex(-1) {
        sz = replaceColumn.size();
    }


    SubMatrix(const SubMatrix& p) : source(nullptr), prev(&p), colIndex(-1) {
        sz = p.size() - 1;
    }

    SubMatrix(const SubMatrix& p, int deletedColumnIndex) : source(nullptr), prev(&p), colIndex(deletedColumnIndex) {
        sz = p.size() - 1;
    }

    int columnIndex() const {
        return colIndex;
    }
    void columnIndex(int index) {
        colIndex = index;
    }

    size_t size() const {
        return sz;
    }

    double index(int row, int col) const {
        if (source != nullptr) {
            if (col == colIndex) {
                return replaceColumn[row];
            }
            else {
                return (*source)[row][col];
            }
        }
        else {
            if (col < colIndex) {
                return prev->index(row + 1, col);
            }
            else {
                return prev->index(row + 1, col + 1);
            }
        }
    }

    double det() const {
        if (sz == 1) {
            return index(0, 0);
        }
        if (sz == 2) {
            return index(0, 0) * index(1, 1) - index(0, 1) * index(1, 0);
        }
        SubMatrix m(*this);
        double det = 0.0;
        int sign = 1;
        for (size_t c = 0; c < sz; ++c) {
            m.columnIndex(c);
            double d = m.det();
            det += index(0, c) * d * sign;
            sign = -sign;
        }
        return det;
    }


    double ParallelDet() const {
        if (sz == 1) {
            return index(0, 0);
        }
        if (sz == 2) {
            return index(0, 0) * index(1, 1) - index(0, 1) * index(1, 0);
        }
        SubMatrix m(*this);
        double det = 0.0;
        int sign = 1;

        concurrent_vector<tuple<SubMatrix, double>> submatrix;



        parallel_for_each(size_t(0), sz, [&](size_t c) {
            m.columnIndex(c);
            double d = m.ParallelDet();
            det += index(0, c) * d * sign;
            sign = -sign;
            });
        return det;
    }
};

std::vector<double> solveParallel(SubMatrix& matrix) {
    double det = matrix.ParallelDet();
    if (det == 0.0) {
        throw std::runtime_error("The determinant is zero.");
    }

    

    std::vector<double> answer(matrix.size());
    for (int i = 0; i < matrix.size(); ++i) {
        matrix.columnIndex(i);
        answer[i] = matrix.ParallelDet() / det;
    }
    return answer;
}


std::vector<double> solveSerial(SubMatrix& matrix) {
    double det = matrix.det();
    if (det == 0.0) {
        throw std::runtime_error("The determinant is zero.");
    }

    std::vector<double> answer(matrix.size());
    for (int i=0; i < matrix.size(); ++i) {
        matrix.columnIndex(i);
        answer[i] = matrix.det() / det;
    }
    return answer;
}

std::vector<double> solveCramer(const std::vector<std::vector<double>>& equations) {
    int size = equations.size();
    if (std::any_of(
        equations.cbegin(), equations.cend(),
        [size](const std::vector<double>& a) { return a.size() != size + 1; }
    )) {
        throw std::runtime_error("Each equation must have the expected size.");
    }

    std::vector<std::vector<double>> matrix(size);
    std::vector<double> column(size);
    parallel_for(0, size, [&](size_t i) {
        column[i] = equations[i][size];
        matrix[i].resize(size);
        for (int c = 0; c < size; ++c) {
            matrix[i][c] = equations[i][c];
        }
        });

    SubMatrix sm(matrix, column);
    return solveParallel(sm);
}



std::vector<double> solveCramerSerial(const std::vector<std::vector<double>>& equations) {
    int size = equations.size();
    if (std::any_of(
        equations.cbegin(), equations.cend(),
        [size](const std::vector<double>& a) { return a.size() != size + 1; }
    )) {
        throw std::runtime_error("Each equation must have the expected size.");
    }

    std::vector<std::vector<double>> matrix(size);
    std::vector<double> column(size);
    for (int r = 0; r < size; ++r) {
        column[r] = equations[r][size];
        matrix[r].resize(size);
        for (int c = 0; c < size; ++c) {
            matrix[r][c] = equations[r][c];
        }
    }

    SubMatrix sm(matrix, column);
    return solveSerial(sm);
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    auto it = v.cbegin();
    auto end = v.cend();

    os << '[';
    if (it != end) {
        os << *it++;
    }
    while (it != end) {
        os << ", " << *it++;
    }

    return os << ']';
}

int main() {
    double start_time, end_time;
    std::vector<std::vector<double>> equations = {
        { 2, -1,  5,  1,  -3},
        { 3,  2,  2, -6, -32},
        { 1,  3,  3, -1, -47},
        { 5, -2, -3,  3,  49},
    };

    /*for (int i = 0; i < 1000; i++){
        auto solution = solveCramerSerial(equations);
        solution = solveCramer(equations);
    }*/
    for (int i = 0; i < 5; i++) {
        start_time = omp_get_wtime();
        auto solution = solveCramerSerial(equations);
        end_time = omp_get_wtime() - start_time;



        std::cout << "Serial time taken in seconds: " << end_time << "s\n";
        std::cout << solution << '\n';

        start_time = omp_get_wtime();
        solution = solveCramer(equations);
        end_time = omp_get_wtime() - start_time;

        std::cout << "Parallel time taken in seconds: " << end_time << "s\n";
        std::cout << solution << '\n';
    }
    return 0;
}