//Matrix opertation libaray for Charto neural network
//create matrix: matrixBuilder([i,j]: matrix dimension, containerValue, theshold: for the dropout)
var matrixBuilder = function (dimention, container, threshold) {
    var matrix = new Array();
    for (var idxI = 0; idxI < dimention[0]; idxI++) {
        var matrixSubUnit = new Array();
        for (var idxJ = 0; idxJ < dimention[1]; idxJ++) {
            if (container) {
                matrixSubUnit.push(container);
            } else {
                var value = Math.random();
                if (threshold) {
                    value = value < threshold ? 0 : 1;
                }
                matrixSubUnit.push(value);
            }
        }
        matrix.push(matrixSubUnit);
    }
    return matrix;
};
/*matrix operation: 
 * .matrix: original matrix,
 * .shape(): print matrix with coressponding dimension
 * .show(): print the matrix with corrsponding demension
 * .T(): transport matrix
 * .dot(M): multiply matrix with M elemently
 * .dot(M, "/"): devide matrix with M elemently
 * .mul(M): multiply matrix with M
 * .add(M): add matrix with M elemently
 * .add(M, "-"): minus matrix with M elemently
 */
function Matrix(originalMatrix) {
    this.matrix = originalMatrix;
    this.shape = function () {
        return [originalMatrix.length, originalMatrix[0].length];
    };
    this.T = function () {
        var Matrix = matrixBuilder([originalMatrix[0].length, originalMatrix.length, 0]);
        for (var i = 0; i < originalMatrix.length; i++) {
            for (var j = 0; j < originalMatrix[0].length; j++) {
                Matrix[j][i] = originalMatrix[i][j];
            }
        }
        return Matrix;
    };
    this.dot = function (matrix, operation) {
        var i1 = originalMatrix.length;
        var j1 = originalMatrix[0].length;
        var i2 = matrix.length;
        var j2 = matrix[0].length;
        if (operation === "/") {
            for (var i = 0; i < i2; i++) {
                for (var j = 0; j < j2; j++) {
                    matrix[i][j] = 1 / matrix[i][j] + Math.pow(1, -12);
                }
            }
        }
        if (i1 !== i2 && j1 !== j2) {
            throw "Error: unable to dot the matrix with wrong dimension";
        } else if (i1 === i2 && j1 === j2) {
            var Matrix = matrixBuilder([i1, j1], 0);
            for (var i = 0; i < i1; i++) {
                for (var j = 0; j < j1; j++) {
                    Matrix[i][j] = originalMatrix[i][j] * matrix[i][j];
                }
            }
        } else {
            if (i2 === 1 || j2 === 1) {
                var Matrix = matrixBuilder([i1, j1], 0);
                for (var i = 0; i < i1; i++) {
                    for (var j = 0; j < j1; j++) {
                        if (i2 === 1 && j2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] * matrix[0][0];
                        } else if (i2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] * matrix[0][j];
                        } else if (j2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] * matrix[i][0];
                        }
                    }
                }
            } else if (i1 === 1 || j1 === 1) {
                var Matrix = matrixBuilder([i2, j2], 0);
                for (var i = 0; i < i2; i++) {
                    for (var j = 0; j < j2; j++) {
                        if (i1 === 1 && j1 === 1) {
                            Matrix[i][j] = originalMatrix[0][0] * matrix[i][j];
                        } else if (i1 === 1) {
                            Matrix[i][j] = originalMatrix[0][j] * matrix[i][j];
                        } else if (j1 === 1) {
                            Matrix[i][j] = originalMatrix[i][0] * matrix[i][j];
                        }
                    }
                }
            } else {
                throw "Error: unable to broadcast the matrix with non-one demension";
            }
        }
        return Matrix;
    };
    this.mul = function (matrix) {
        var i1 = originalMatrix.length;
        var j1 = originalMatrix[0].length;
        var i2 = matrix.length;
        var j2 = matrix[0].length;
        if (j1 !== i2) {
            throw "Error: unable to multiply the matrix with wrong dimension";
        } else {
            var Matrix = matrixBuilder([i1, j2], 0);
            for (var i = 0; i < i1; i++) {
                for (var j = 0; j < j2; j++) {
                    for (var k = 0; k < j1; k++) {
                        Matrix[i][j] += originalMatrix[i][k] * matrix[k][j];
                    }
                }
            }
        }
        return Matrix;
    };
    this.add = function (matrix, operation) {
        var i1 = originalMatrix.length;
        var j1 = originalMatrix[0].length;
        var i2 = matrix.length;
        var j2 = matrix[0].length;
        if (operation === "-") {
            for (var i = 0; i < i2; i++) {
                for (var j = 0; j < j2; j++) {
                    matrix[i][j] = -matrix[i][j];
                }
            }
        }
        if (i1 !== i2 && j1 !== j2) {
            throw "Error: unable to add the matrix with different dimension";
        } else if (i1 === i2 && j1 === j2) {
            var Matrix = matrixBuilder([i1, j1], 0);
            for (var i = 0; i < i1; i++) {
                for (var j = 0; j < j1; j++) {
                    Matrix[i][j] = originalMatrix[i][j] + matrix[i][j];
                }
            }
        } else {
            if (i2 === 1 || j2 === 1) {
                var Matrix = matrixBuilder([i1, j1], 0);
                for (var i = 0; i < i1; i++) {
                    for (var j = 0; j < j1; j++) {
                        if (i2 === 1 && j2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] + matrix[0][0];
                        } else if (i2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] + matrix[0][j];
                        } else if (j2 === 1) {
                            Matrix[i][j] = originalMatrix[i][j] + matrix[i][0];
                        }
                    }
                }
            } else if (i1 === 1 || j1 === 1) {
                var Matrix = matrixBuilder([i2, j2], 0);
                for (var i = 0; i < i2; i++) {
                    for (var j = 0; j < j2; j++) {
                        if (i1 === 1 && j1 === 1) {
                            Matrix[i][j] = originalMatrix[0][0] + matrix[i][j];
                        } else if (i1 === 1) {
                            Matrix[i][j] = originalMatrix[0][j] + matrix[i][j];
                        } else if (j1 === 1) {
                            Matrix[i][j] = originalMatrix[i][0] + matrix[i][j];
                        }
                    }
                }
            } else {
                throw "Error: unable to broadcast the matrix with non-one demension";
            }
        }
        return Matrix;
    };
    this.show = function () {
        var i1 = originalMatrix.length;
        var j1 = originalMatrix[0].length;
        for (var i = 0; i < i1; i++) {
            print(originalMatrix[i]);
        }
    };
}

var matrix = matrixBuilder([2, 4], "", 0.5);
print(matrix);
var matrix2 = matrixBuilder([2, 1]);
matrix1 = new Matrix(matrix);
matrix1.show();
print(matrix1.dot(matrix2, "/"));