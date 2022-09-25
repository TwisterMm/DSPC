import threading
import time


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def determinantOfMatrix(mat):
 
    ans = (mat[0][0] * (mat[1][1] * mat[2][2] -
                        mat[2][1] * mat[1][2]) -
           mat[0][1] * (mat[1][0] * mat[2][2] -
                        mat[1][2] * mat[2][0]) +
           mat[0][2] * (mat[1][0] * mat[2][1] -
                        mat[1][1] * mat[2][0]))
    return ans
 
# This function finds the solution of system of
# linear equations using cramer's rule
def findSolutionParallel(coeff):
 
    # Matrix d using coeff as given in
    # cramer's rule
    d = [[coeff[0][0], coeff[0][1], coeff[0][2]],
         [coeff[1][0], coeff[1][1], coeff[1][2]],
         [coeff[2][0], coeff[2][1], coeff[2][2]]]
     
    # Matrix d1 using coeff as given in
    # cramer's rule
    d1 = [[coeff[0][3], coeff[0][1], coeff[0][2]],
          [coeff[1][3], coeff[1][1], coeff[1][2]],
          [coeff[2][3], coeff[2][1], coeff[2][2]]]
     
    # Matrix d2 using coeff as given in
    # cramer's rule
    d2 = [[coeff[0][0], coeff[0][3], coeff[0][2]],
          [coeff[1][0], coeff[1][3], coeff[1][2]],
          [coeff[2][0], coeff[2][3], coeff[2][2]]]
     
    # Matrix d3 using coeff as given in
    # cramer's rule
    d3 = [[coeff[0][0], coeff[0][1], coeff[0][3]],
          [coeff[1][0], coeff[1][1], coeff[1][3]],
          [coeff[2][0], coeff[2][1], coeff[2][3]]]
 
    # Calculating Determinant of Matrices
    # d, d1, d2, d3
    t1 = ThreadWithReturnValue(target=determinantOfMatrix, args=(d,))
    t2 = ThreadWithReturnValue(target=determinantOfMatrix, args=(d1,))
    t3 = ThreadWithReturnValue(target=determinantOfMatrix, args=(d2,))
    t4 = ThreadWithReturnValue(target=determinantOfMatrix, args=(d3,))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    D = t1.join()
    D1 = t2.join()
    D2 = t3.join()
    D3 = t4.join()

    # D = determinantOfMatrix(d)
    # D1 = determinantOfMatrix(d1)
    # D2 = determinantOfMatrix(d2)
    # D3 = determinantOfMatrix(d3)
     
    print("D is : ", D)
    print("D1 is : ", D1)
    print("D2 is : ", D2)
    print("D3 is : ", D3)
 
    # Case 1
    if (D != 0):
       
        # Coeff have a unique solution.
        # Apply Cramer's Rule
        x = D1 / D
        y = D2 / D
         
        # calculating z using cramer's rule
        z = D3 / D 
         
        print("Value of x is : ", x)
        print("Value of y is : ", y)
        print("Value of z is : ", z)
 
    # Case 2
    else:
        if (D1 == 0 and D2 == 0 and
            D3 == 0):
            print("Infinite solutions")
        elif (D1 != 0 or D2 != 0 or
              D3 != 0):
            print("No solutions")

def findSolution(coeff):
 
    # Matrix d using coeff as given in
    # cramer's rule
    #initialise determinant
    
    # for _ in range(len(coeff)):
    #     for row in range(len(coeff[0])):
    #         for col in range(len(coeff[0])):
    #             d[row][col] = coeff[row][col]

    d = [[coeff[0][0], coeff[0][1], coeff[0][2]],
         [coeff[1][0], coeff[1][1], coeff[1][2]],
         [coeff[2][0], coeff[2][1], coeff[2][2]]]
     
    # Matrix d1 using coeff as given in
    # cramer's rule
    d1 = [[coeff[0][3], coeff[0][1], coeff[0][2]],
          [coeff[1][3], coeff[1][1], coeff[1][2]],
          [coeff[2][3], coeff[2][1], coeff[2][2]]]
     
    # Matrix d2 using coeff as given in
    # cramer's rule
    d2 = [[coeff[0][0], coeff[0][3], coeff[0][2]],
          [coeff[1][0], coeff[1][3], coeff[1][2]],
          [coeff[2][0], coeff[2][3], coeff[2][2]]]
     
    # Matrix d3 using coeff as given in
    # cramer's rule
    d3 = [[coeff[0][0], coeff[0][1], coeff[0][3]],
          [coeff[1][0], coeff[1][1], coeff[1][3]],
          [coeff[2][0], coeff[2][1], coeff[2][3]]]
 
    # Calculating Determinant of Matrices
    # d, d1, d2, d3
    D = determinantOfMatrix(d)
    D1 = determinantOfMatrix(d1)
    D2 = determinantOfMatrix(d2)
    D3 = determinantOfMatrix(d3)
     
    print("D is : ", D)
    print("D1 is : ", D1)
    print("D2 is : ", D2)
    print("D3 is : ", D3)
 
    # Case 1
    if (D != 0):
       
        # Coeff have a unique solution.
        # Apply Cramer's Rule
        x = D1 / D
        y = D2 / D
         
        # calculating z using cramer's rule
        z = D3 / D 
         
        print("Value of x is : ", x)
        print("Value of y is : ", y)
        print("Value of z is : ", z)
 
    # Case 2
    else:
        if (D1 == 0 and D2 == 0 and
            D3 == 0):
            print("Infinite solutions")
        elif (D1 != 0 or D2 != 0 or
              D3 != 0):
            print("No solutions")

# Driver Code
if __name__ == "__main__":
 
    # storing coefficients of linear
    # equations in coeff matrix
    coeff = [[2, -1,  5,  1],
             [3,  2,  2, -6],
             [1,  3,  3, -1],] 
    

    start_time = time.time()
    findSolution(coeff)
    total_time = time.time() - start_time
    print(f"Total time taken: %.5f"% total_time)

    start_time = time.time()
    findSolutionParallel(coeff)
    total_time = time.time() - start_time
    print(f"Parallel total time taken: %.5f"% total_time)