#include <algorithm>
#include <iostream>
#include <vector>
#include "omp.h"
#include <chrono>

std::chrono::steady_clock::time_point begin;
std::chrono::steady_clock::time_point end;
int id, nthrds;

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
};

std::vector<double> solve(SubMatrix& matrix) {
    double det = matrix.det();
    if (det == 0.0) {
        throw std::runtime_error("The determinant is zero.");
    }

    std::vector<double> answer(matrix.size());
#pragma omp parallel
    {
        id = omp_get_thread_num();
        matrix.columnIndex(id);
        answer[id] = matrix.det() / det;
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
    omp_set_num_threads(size);
    std::vector<std::vector<double>> matrix(size);
    std::vector<double> column(size);
    begin = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        id = omp_get_thread_num();
        column[id] = equations[id][size];
        matrix[id].resize(size);
        for (int c = 0; c < size; ++c) {
            matrix[id][c] = equations[id][c];
        }
    }
    SubMatrix sm(matrix, column);
    
    return solve(sm);
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
    std::vector<std::vector<double>> equations = {
        { 2, -1,  5,  1,  -3},
        { 3,  2,  2, -6, -32},
        { 1,  3,  3, -1, -47},
        { 5, -2, -3,  3,  49},
    };

    auto solution = solveCramer(equations);
    end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " micro seconds" << std::endl;
    std::cout << solution << '\n';

    return 0;
}