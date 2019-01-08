//Matrix opertation libaray for JS
//Arthur: L.Wu
var ML = ML || {};
(function (ml) {
    var NN = function (config) {
        config = config || {};
        if (!config.layer_dims) {
            throw "Error: please specify the NN layer structure in an array";
        }
        if (!config.iterations) {
            config.iterations = 1000;
        }
        if (!config.learning_rate) {
            config.learning_rate = 0.001;
        }
        if (!config.lambda) {
            config.lambda = 0.1;
        }
        if (!config.init_method) {
            //default: random inintialization
            config.init_method = null;
        }
        if (!config.print_cost) {
            config.print_cost = true;
        }
        this.layer_dims = config.layer_dims;
        this.iterations = config.iterations;
        this.learning_rate = new Matrix([[config.learning_rate]]);
        this.lambda = config.lambda;
        this.init_method = config.init_method;
        this.print_cost = config.print_cost;
    };
    function broadcast(matrix1, matrix2) {
        matrix1 = new Matrix(matrix1);
        matrix2 = new Matrix(matrix2);
        var i1 = matrix1.shape()[0];
        var j1 = matrix1.shape()[1];
        var i2 = matrix2.shape()[0];
        var j2 = matrix2.shape()[1];
        matrix1 = matrix1.matrix;
        matrix2 = matrix2.matrix;
        //if two dimensions are the same
        if (i1 === i2 && j1 === j2) {
            return [matrix1, matrix2];
        }
        //if one dimensions is the same, four cases
        // case1: (3, 1); (3, 3)
        else if (i1 === i2 && j1 === 1) {
            var matrixContainer = new Matrix();
            matrixContainer.create([i2, j2], 0);
            for (var i = 0; i < i2; i++) {
                for (var j = 0; j < j2; j++) {
                    matrixContainer.matrix[i][j] = matrix1[i][0];
                }
            }
            matrix1 = matrixContainer.matrix;
        }
        // case2: (3, 3); (3, 1)
        else if (i1 === i2 && j2 === 1) {
            var matrixContainer = new Matrix();
            matrixContainer.create([i1, j1], 0);
            for (var i = 0; i < i1; i++) {
                for (var j = 0; j < j1; j++) {
                    matrixContainer.matrix[i][j] = matrix2[i][0];
                }
            }
            matrix2 = matrixContainer.matrix;
        }
        // case3: (1, 3); (3, 3)
        else if (j1 === j2 && i1 === 1) {
            var matrixContainer = new Matrix();
            matrixContainer.create([i2, j2], 0);
            for (var i = 0; i < i2; i++) {
                for (var j = 0; j < j2; j++) {
                    matrixContainer.matrix[i][j] = matrix1[0][j];
                }
            }
            matrix1 = matrixContainer.matrix;
        }
        // case4: (3, 3); (1, 3)
        else if (j1 === j2 && i2 === 1) {
            var matrixContainer = new Matrix();
            matrixContainer.create([i1, j1], 0);
            for (var i = 0; i < i1; i++) {
                for (var j = 0; j < j1; j++) {
                    matrixContainer.matrix[i][j] = matrix2[0][j];
                }
            }
            matrix2 = matrixContainer.matrix;
        }
        //if two dimensions are different, three cases
        else if (i1 !== i2 && j1 !== j2) {
            // case1: (1, 1), (3, 3)
            if (i1 === j1 && j1 === 1) {
                var matrixContainer = new Matrix();
                matrixContainer.create([i2, j2], matrix1[0][0]);
                matrix1 = matrixContainer.matrix;
            }
            // case2: (3, 3), (1, 1)
            else if (i2 === j2 && j2 === 1) {
                var matrixContainer = new Matrix();
                matrixContainer.create([i1, j1], matrix2[0][0]);
                matrix2 = matrixContainer.matrix;
            }
            // case2: (1, 2), (3, 4)
            else {
                throw "Error: unable to boardcast the matrix with dimension (" + i1 + "," + j1 + ") and (" + i2 + "," + j2 + ")";
            }
        } else {
            throw "Error: failed to boardcast the matrix with dimension (" + i1 + "," + j1 + ") and (" + i2 + "," + j2 + ")";
        }
        return [matrix1, matrix2];
    }
    function Matrix(inputMatrix) {
        if (inputMatrix) {
            try {
                var temp = inputMatrix.matrix;
                if (temp) {
                    this.matrix = temp;
                } else {
                    throw ".matrix is undefined";
                }
            } catch (e) {
                this.matrix = inputMatrix;
            }
        }
        this.create = function (dimention, container, threshold) {
            var Matrix = new Array();
            for (var idxI = 0; idxI < dimention[0]; idxI++) {
                var matrixSubUnit = new Array();
                for (var idxJ = 0; idxJ < dimention[1]; idxJ++) {
                    if (container === +container) {
                        matrixSubUnit.push(container);
                    } else {
                        var value = Math.random();
                        if (threshold) {
                            value = value < threshold ? 0 : 1;
                        }
                        matrixSubUnit.push(value);
                    }
                }
                Matrix.push(matrixSubUnit);
            }
            this.matrix = Matrix;
        };
        this.shape = function () {
            return [this.matrix.length, this.matrix[0].length];
        };
        this.T = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([width, length, 0]);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[j][i] = this.matrix[i][j];
                }
            }
            return matrixContainer;
        };
        this.inv = function () {
            //if the matrix isn't square: exit (error)
            if (this.matrix.length !== this.matrix[0].length) {
                throw "Error: the matrix isn't square";
            }
            //create the identity matrix (matrixContainer), and a dummy (dummyMatrix) of the original
            var i = 0, ii = 0, j = 0, dim = this.matrix.length, element = 0, t = 0;
            var matrixContainer = new Matrix();
            matrixContainer.create([dim, dim], 0);
            var dummyMatrix = [];
            for (i = 0; i < dim; i += 1) {
                dummyMatrix[dummyMatrix.length] = [];
                for (j = 0; j < dim; j += 1) {
                    if (i === j) {
                        matrixContainer.matrix[i][j] = 1;
                    } else {
                        matrixContainer.matrix[i][j] = 0;
                    }
                    dummyMatrix[i][j] = this.matrix[i][j];
                }
            }
            // Perform elementary row operations
            for (i = 0; i < dim; i += 1) {
                element = dummyMatrix[i][i];
                if (element === 0) {
                    for (ii = i + 1; ii < dim; ii += 1) {
                        if (dummyMatrix[ii][i] !== 0) {
                            for (j = 0; j < dim; j++) {
                                element = dummyMatrix[i][j];
                                dummyMatrix[i][j] = dummyMatrix[ii][j];
                                dummyMatrix[ii][j] = element;
                                element = matrixContainer.matrix[i][j];
                                matrixContainer.matrix[i][j] = matrixContainer.matrix[ii][j];
                                matrixContainer.matrix[ii][j] = element;
                            }
                            break;
                        }
                    }
                    element = dummyMatrix[i][i];
                    //if the element is still zero: exit (error)
                    if (element === 0) {
                        throw "Error: the matrix is not invertable";
                    }
                }
                for (j = 0; j < dim; j++) {
                    dummyMatrix[i][j] = dummyMatrix[i][j] / element;
                    matrixContainer.matrix[i][j] = matrixContainer.matrix[i][j] / element;
                }
                for (ii = 0; ii < dim; ii++) {
                    if (ii === i) {
                        continue;
                    }
                    element = dummyMatrix[ii][i];
                    for (j = 0; j < dim; j++) {
                        dummyMatrix[ii][j] -= element * dummyMatrix[i][j];
                        matrixContainer.matrix[ii][j] -= element * matrixContainer.matrix[i][j];
                    }
                }
            }
            return matrixContainer;
        };
        this.log = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width, 0]);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = Math.log(this.matrix[i][j]);
                }
            }
            return matrixContainer;
        };
        this.exp = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width, 0]);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = Math.exp(this.matrix[i][j]);
                }
            }
            return matrixContainer;
        };
        this.sum = function (axis) {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            if (axis === 0) {
                var matrixContainer = new Matrix();
                matrixContainer.create([1, width], 0);
                for (var j = 0; j < width; j++) {
                    for (var i = 0; i < length; i++) {
                        matrixContainer.matrix[0][j] += this.matrix[i][j];
                    }
                }
            } else if (axis === 1) {
                var matrixContainer = new Matrix();
                matrixContainer.create([length, 1], 0);
                for (var i = 0; i < length; i++) {
                    for (var j = 0; j < width; j++) {
                        matrixContainer.matrix[i][0] += this.matrix[i][j];
                    }
                }
            } else {
                var matrixContainer = new Matrix();
                matrixContainer.create([1, 1], 0);
                for (var i = 0; i < length; i++) {
                    for (var j = 0; j < width; j++) {
                        matrixContainer[0][0] += this.matrix[i][j];
                    }
                }
            }
            return matrixContainer;
        };
        this.dot = function (matrix) {
            var i1 = this.matrix.length;
            var j1 = this.matrix[0].length;
            try {
                var i2 = matrix.length;
                var j2 = matrix[0].length;
            } catch (e) {
                matrix = matrix.matrix;
                var i2 = matrix.length;
                var j2 = matrix[0].length;
            }
            if (j1 !== i2) {
                throw "Error: unable to dot the matrix with dimension (" + i1 + "," + j1 + ") and (" + i2 + "," + j2 + ")";
            } else {
                var matrixContainer = new Matrix();
                matrixContainer.create([i1, j2], 0);
                for (var i = 0; i < i1; i++) {
                    for (var j = 0; j < j2; j++) {
                        for (var k = 0; k < j1; k++) {
                            matrixContainer.matrix[i][j] += this.matrix[i][k] * matrix[k][j];
                        }
                    }
                }
            }
            return matrixContainer;
        };
        this.mul = function (matrix) {
            matrix = broadcast(this.matrix, matrix);
            var length = matrix[0].length;
            var width = matrix[0][0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = matrix[0][i][j] * matrix[1][i][j];
                }
            }
            return matrixContainer;
        };
        this.div = function (matrix) {
            matrix = broadcast(this.matrix, matrix);
            var length = matrix[0].length;
            var width = matrix[0][0].length;
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrix[1][i][j] = matrix[1][i][j] + Math.pow(0.1, 10);
                }
            }
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = matrix[0][i][j] / matrix[1][i][j];
                }
            }
            return matrixContainer;
        };
        this.add = function (matrix) {
            matrix = broadcast(this.matrix, matrix);
            var length = matrix[0].length;
            var width = matrix[0][0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = matrix[0][i][j] + matrix[1][i][j];
                }
            }
            return matrixContainer;
        };
        this.minus = function (matrix) {
            matrix = broadcast(this.matrix, matrix);
            var length = matrix[0].length;
            var width = matrix[0][0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = matrix[0][i][j] - matrix[1][i][j];
                }
            }
            return matrixContainer;
        };
        this.show = function () {
            var dim = this.matrix.length;
            for (var i = 0; i < dim; i++) {
                print(this.matrix[i]);
            }
        };
    }
//return cache["W"/"b" + No.layer]
    function initialize_parameters(layer_dims, init_method) {
        var parameters = new Object();
        var L = layer_dims.length;
        for (var idx = 1; idx < L; idx++) {
            parameters["W" + String(idx)] = new Matrix();
            parameters["b" + String(idx)] = new Matrix();
            parameters["W" + String(idx)].create([layer_dims[idx], layer_dims[idx - 1]]);
            parameters["b" + String(idx)].create([layer_dims[idx], 1], 0);
            if(init_method === "He"){
                parameters["W" + String(idx)].mul([[Math.pow(2/layer_dims[idx - 1],0.5)]]);
            }
        }
        return parameters;
    }
//return A, linear_cache, activation_cache
    function linear_activation_forward(A_prev, W, b, activation) {
        //return Z, cache["A"/"W"/"b"]
        var linear_forward = function (A, W, b) {
            var cache = new Object();
            var Z = W.dot(A).add(b);
            cache["A"] = new Matrix(A);
            cache["W"] = new Matrix(W);
            cache["b"] = new Matrix(b);
            return {
                Z: Z,
                cache: cache};
        };
        var sigmiod = function (inputMatrix) {
            var Z = new Matrix(inputMatrix);
            Z = Z.matrix;
            var i = Z.length;
            var j = Z[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([i, j], 0);
            for (var idxI = 0; idxI < i; idxI++) {
                for (var idxJ = 0; idxJ < j; idxJ++) {
                    matrixContainer.matrix[idxI][idxJ] = 1 / (1 + Math.pow(Math.E, -Z[idxI][idxJ]));
                }
            }
            return {
                A: matrixContainer,
                cache: inputMatrix};
        };
        var relu = function (inputMatrix) {
            var Z = new Matrix(inputMatrix);
            var Z = Z.matrix;
            var i = Z.length;
            var j = Z[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([i, j], 0);
            for (var idxI = 0; idxI < i; idxI++) {
                for (var idxJ = 0; idxJ < j; idxJ++) {
                    matrixContainer.matrix[idxI][idxJ] = Z[idxI][idxJ] > 0 ? Z[idxI][idxJ] : 0;
                }
            }
            return {
                A: matrixContainer,
                cache: inputMatrix};
        };
        var activationForward;
        var linearForward = linear_forward(A_prev, W, b);
        if (activation === "sigmoid") {
            activationForward = sigmiod(linearForward.Z);
        } else if (activation === "relu") {
            activationForward = relu(linearForward.Z);
        }
        return {
            A: activationForward.A,
            linear_cache: linearForward.cache,
            activation_cache: activationForward.cache
        };
    }
//return A, cache["linear"/"activation" + No.layer]
    function L_model_forward(X, parameters) {
        var cache = new Object();
        var A = X;
        var L = 0;
        for (var i in parameters) {
            L++;
        }
        L = Math.floor(L / 2);
        for (var l = 1; l < L; l++) {
            var A_prev = A;
            var activationForward = linear_activation_forward(A_prev, parameters["W" + String(l)], parameters["b" + String(l)], "relu");
            A = activationForward.A;
            cache["linear" + String(l)] = activationForward.linear_cache;
            cache["activation" + String(l)] = activationForward.activation_cache;
        }
        activationForward = linear_activation_forward(A, parameters["W" + String(L)], parameters["b" + String(L)], activation = "sigmoid");
        cache["linear" + String(L)] = activationForward.linear_cache;
        cache["activation" + String(L)] = activationForward.activation_cache;
        return{A: activationForward.A,
            cache: cache
        };
    }
//return cost value
    function compute_cost(AL, Y) {
        var m = [[Y[0].length]];
        var one = new Matrix([[1]]);
        var Y = new Matrix(Y);
        if (AL.shape()[0] !== Y.shape()[0]) {
            throw "Error: unable to compute the cost with the dimension of the input matrixes of (" + AL.shape()[0] + "," + AL.shape()[1] + ") and (" + Y.shape()[0] + "," + Y.shape()[1] + ")";
        }
        var cost = one.div(m).mul(Y.mul(AL.log()).add(one.minus(Y).mul(one.minus(AL).log())));
        return cost.matrix[0][0];
    }
//return dW, db, dA_prev
    function linear_activation_backward(dA, linear_cache, activation_cache, activation) {
        //return dW, db, dA_prev
        var linear_backward = function (dZ, cache) {
            var A_prev = cache["A"];
            var W = cache["W"];
            var b = cache["b"];
            var m = [[A_prev.matrix.length]];
            var one = new Matrix([[1]]);
            return {
                dW: one.div(m).mul(dZ.dot(A_prev.T())),
                db: one.div(m).mul(dZ.sum(1)),
                dA_prev: W.T().dot(dZ)
            };
        };
        var relu_backward = function (dA, cache) {
            var length = cache.shape()[0];
            var width = cache.shape()[1];
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = (cache.matrix[i][j] <= 0 ? 0 : dA.matrix[i][j]);
                }
            }
            return  matrixContainer;
        };
        var sigmoid_backward = function (dA, cache) {
            var z = cache.matrix;
            var length = cache.shape()[0];
            var width = cache.shape()[1];
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    var s = 1 / (1 + Math.exp(-z[i][j]));
                    matrixContainer.matrix[i][j] = dA.matrix[i][j] * s * (1 - s);
                }
            }
            return  matrixContainer;
        };
        var dZ, result;
        if (activation === "relu") {
            dZ = relu_backward(dA, activation_cache);
        } else if (activation === "sigmoid") {
            dZ = sigmoid_backward(dA, activation_cache);
        }
        result = linear_backward(dZ, linear_cache);
        return {
            dW: result.dW,
            db: result.db,
            dA_prev: result.dA_prev
        };
    }
//return grad["dA"/"dW"/"dB" + No.layer]
    function L_model_backward(AL, Y, cache) {
        var Y = new Matrix(Y);
        var grad = new Object();
        var m = AL.shape()[0];
        var one = new Matrix([[1]]);
        var L = 0;
        for (var i in cache) {
            L++;
        }
        L = Math.floor(L / 2);
        var dAL = ((one.minus(Y)).div(one.minus(AL))).minus(Y.div(AL));
        var linear_cache = cache["linear" + String(L)];
        var activation_cache = cache["activation" + String(L)];
        var result = linear_activation_backward(dAL, linear_cache, activation_cache, "sigmoid");
        grad["dA" + String(L - 1)] = result.dA_prev;
        grad["dW" + String(L)] = result.dW;
        grad["db" + String(L)] = result.db;
        for (var l = L - 1; l > 0; l--) {
            var result = linear_activation_backward(grad["dA" + String(l)], cache["linear" + String(l)], cache["activation" + String(l)], "relu");
            grad["dA" + String(l - 1)] = result.dA_prev;
            grad["dW" + String(l)] = result.dW;
            grad["db" + String(l)] = result.db;
        }
        return grad;
    }
//return updateParameters["W"/"b" + No.layer]
    function update_parameters(parameters, grads, learning_rate) {
        var updateParameters = new Object();
        //learning_rate = new Matrix([[learning_rate]]);
        var L = 0;
        for (var i in parameters) {
            L++;
        }
        L = Math.floor(L / 2);
        for (var l = 1; l <= L; l++) {

            updateParameters["W" + String(l)] = parameters["W" + String(l)].minus(learning_rate.mul(grads["dW" + String(l)]));
            updateParameters["b" + String(l)] = parameters["b" + String(l)].minus(learning_rate.mul(grads["db" + String(l)]));
        }
        return updateParameters;
    }
    NN.prototype.fit = function (X, Y) {
        var parameters = initialize_parameters(this.layer_dims, this.init_method);
        for (var i = 0; i < this.iterations; i++) {
            var forward = L_model_forward(X, parameters);
            var cost = compute_cost(forward.A, Y);
            var grads = L_model_backward(forward.A, Y, forward.cache);
            parameters = update_parameters(parameters, grads, this.learning_rate);
            if (this.print_cost && i % 100 === 0) {
                print("Cost after iteration " + i + ": " + cost);
                //cost.show();
            }
        }
    };
    ml.NN = NN;
})(ML);

X = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]];
Y = [[0.95, 0.95, 0.95, 0.95, 0.95, 0.95]];
var NN = new ML.NN({
    layer_dims: [6, 5, 2, 1],
    iterations: 1000,
    learning_rate: 0.01,
    lambda: 0,
    init_method: "He",
    print_cost: true
});
NN.fit(X, Y);

