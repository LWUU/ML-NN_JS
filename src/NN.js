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
            config.lambda = 0;
        }
        if (!config.keep_prob) {
            config.keep_prob = 1;
        }
        if (!config.init_method) {
            //default: random inintialization
            config.init_method = null;
        }
        if (!config.print_cost) {
            config.print_cost = false;
        }
        if (!config.gradient_check) {
            config.gradient_check = false;
        }
        this.layer_dims = config.layer_dims;
        this.iterations = config.iterations;
        this.learning_rate = new Matrix([[config.learning_rate]]);
        this.lambda = new Matrix([[config.lambda]]);
        this.keep_prob = config.keep_prob;
        this.init_method = config.init_method;
        this.print_cost = config.print_cost;
        this.gradient_check = config.gradient_check;
        this.layer_num = config.layer_dims.length;
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
                            value = value < threshold ? 1 : 0;
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
            matrixContainer.create([width, length], 0);
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
        this.norm = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([1, 1], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[0][0] += Math.pow(this.matrix[i][j], 2);
                }
            }
            matrixContainer.matrix[0][0] = Math.pow(matrixContainer.matrix[0][0], 0.5);
            return matrixContainer;
        };
        this.toVector = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length * width, 1], 0);
            var counter = 0;
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[counter][0] = this.matrix[i][j];
                    counter++;
                }
            }
            return matrixContainer;
        };
        this.log = function () {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
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
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = Math.exp(this.matrix[i][j]);
                }
            }
            return matrixContainer;
        };
        this.pow = function (power) {
            var length = this.matrix.length;
            var width = this.matrix[0].length;
            var matrixContainer = new Matrix();
            matrixContainer.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    matrixContainer.matrix[i][j] = Math.pow(this.matrix[i][j], power);
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
                        matrixContainer.matrix[0][0] += this.matrix[i][j];
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
            parameters["b" + String(idx)].create([layer_dims[idx], 1]);
            if (init_method === "He") {
                parameters["W" + String(idx)].mul([[Math.pow(2 / layer_dims[idx - 1], 0.5)]]);
            }
        }
        return parameters;
    }
    //return A, cache["W"/"b"/"Z"/"A"/"D"]
    function linear_activation_forward(A_prev, W, b, activation, keep_prob) {
        //return Z, cache["D"]
        var linear_forward = function (A, W, b, keep_prob) {
            var cache = new Object();
            if (keep_prob < 1) {
                var D = new Matrix();
                D.create(A.shape(), "", keep_prob);
                A = A.mul(D).div([[keep_prob]]);
                cache["D"] = D;
            }
            var Z = W.dot(A).add(b);
            return {
                Z: Z,
                cache: cache};
        };
        //return A
        var sigmiod = function (Z) {
            var length = Z.shape()[0];
            var width = Z.shape()[1];
            var A = new Matrix();
            A.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    A.matrix[i][j] = 1 / (1 + Math.exp(-Z.matrix[i][j]));
                }
            }
            return A;
        };
        //return A
        var relu = function (Z) {
            var length = Z.shape()[0];
            var width = Z.shape()[1];
            var A = new Matrix();
            A.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    A.matrix[i][j] = Math.max(Z.matrix[i][j], 0.01);
                }
            }
            return A;
        };
        var cache = new Object();
        var linearForward = linear_forward(A_prev, W, b, keep_prob);
        var Z = linearForward.Z;
        if (activation === "sigmoid") {
            var A = sigmiod(Z);
        } else if (activation === "relu") {
            A = relu(Z);
        }
        cache["A"] = A_prev;
        cache["W"] = W;
        cache["b"] = b;
        cache["Z"] = Z;
        cache["D"] = linearForward.cache["D"];
        return {
            A: A,
            cache: cache
        };
    }
    //return A, cache[No.layer]
    function L_model_forward(X, parameters, layer_num, keep_prob) {
        var cache = new Array();
        var A_prev = X;
        var L = layer_num - 1;
        for (var l = 1; l < L; l++) {
            var forward = linear_activation_forward(A_prev, parameters["W" + String(l)], parameters["b" + String(l)], "relu", keep_prob);
            A_prev = forward.A;
            cache[l] = forward.cache;
        }
        forward = linear_activation_forward(A_prev, parameters["W" + String(L)], parameters["b" + String(L)], "sigmoid", keep_prob);
        cache[L] = forward.cache;
        return{A: forward.A,
            cache: cache
        };
    }
    //return cost value
    function compute_cost(AL, Y, lambda, parameters, layer_num) {
        var m = Y.shape()[1];
        var one = new Matrix([[1]]);
        if (AL.shape()[0] !== Y.shape()[0]) {
            throw "Error: unable to compute the cost with the dimension of the input matrixes of (" + AL.shape()[0] + "," + AL.shape()[1] + ") and (" + Y.shape()[0] + "," + Y.shape()[1] + ")";
        }
        var cost = (Y.mul(AL.log()).add(one.minus(Y).mul(one.minus(AL).log()))).sum().mul([[-1 / m]]);
        //var cost = Y.minus(AL).mul([[-1 / m]]);
        if (lambda > 0) {
            var sum = 0;
            for (var l = 1; l < layer_num; l++) {
                sum += parameters["W" + String(l)].pow(2).sum().matrix[0][0];
            }
            var L2_cost = lambda.div([[2 * m]]).mul([[sum]]);
            cost = cost.add(L2_cost);
        }
        return cost.matrix[0][0];
    }
    //return dW, db, dA_prev
    function linear_activation_backward(dA, cache, activation, lambda, keep_prob) {
        //return grad["dA"/"dZ"/"dW"/"db"]
        var linear_backward = function (dZ, cache, lambda, keep_prob) {
            var grad = new Object();
            var A_prev = cache["A"];
            var W = cache["W"];
            var b = cache["b"];
            var m = A_prev.shape()[1];
            var dW = dZ.dot(A_prev.T()).div([[m]]);
            var db = dZ.sum(1).div([[m]]);
            var dA = W.T().dot(dZ);
            if (lambda.matrix[0][0] > 0) {
                var dW = dW.add(lambda.div([[m]]).mul(W));
            }
            if (keep_prob < 1) {
                dA = dA.mul(cache["D"]).div([[keep_prob]]);
            }
            grad["dZ"] = dZ;
            grad["dW"] = dW;
            grad["dA"] = dA;
            grad["db"] = db;
            return grad;
        };
        //return dZ
        var relu_backward = function (dA, Z) {
            var length = Z.shape()[0];
            var width = Z.shape()[1];
            var dZ = new Matrix();
            dZ.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    dZ.matrix[i][j] = Z.matrix[i][j] < 0 ? 0.01 * dA.matrix[i][j] : dA.matrix[i][j];
                }
            }
            return  dZ;
        };
        //return dZ
        var sigmoid_backward = function (dA, Z) {
            var length = Z.shape()[0];
            var width = Z.shape()[1];
            var dZ = new Matrix();
            dZ.create([length, width], 0);
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    var s = 1 / (1 + Math.exp(-Z.matrix[i][j]));
                    dZ.matrix[i][j] = dA.matrix[i][j] * s * (1 - s);
                }
            }
            return  dZ;
        };
        if (activation === "relu") {
            var dZ = relu_backward(dA, cache["Z"]);
        } else if (activation === "sigmoid") {
            dZ = sigmoid_backward(dA, cache["Z"]);
        }
        var grad = linear_backward(dZ, cache, lambda, keep_prob);
        return grad;
    }
    //return grad[No.layer]
    function L_model_backward(AL, Y, cache, layer_dims, lambda, keep_prob) {
        var Y = new Matrix(Y);
        var grad = new Array();
        var L = layer_dims - 1;
        var dAL = Y.minus([[1]]).div(AL.minus([[1]])).minus(Y.div(AL));
        grad[L] = linear_activation_backward(dAL, cache[L], "sigmoid", lambda, keep_prob);
        for (var l = L - 1; l > 0; l--) {
            grad[l] = linear_activation_backward(grad[l + 1]["dA"], cache[l], "relu", lambda, keep_prob);
        }
        return grad;
    }
    //return updateParameters["W"/"b" + No.layer]
    function update_parameters(parameters, grads, learning_rate, layer_num) {
        var updateParameters = new Object();
        for (var l = 1; l < layer_num; l++) {
            updateParameters["W" + String(l)] = parameters["W" + String(l)].minus(learning_rate.mul(grads[l]["dW"]));
            updateParameters["b" + String(l)] = parameters["b" + String(l)].minus(learning_rate.mul(grads[l]["db"]));
        }
        return updateParameters;
    }
    function gradient_check(parameters, grads, layer_num, lambda, X, Y) {
        var gradient_prob = new Array();
        for (var item in parameters) {
            var length = parameters[item].shape()[0];
            var width = parameters[item].shape()[1];
            //print(item)
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    var parametersDummy = parameters;
                    parametersDummy[item].matrix[i][j] += 1e-4;
                    var forwardPlus = L_model_forward(X, parametersDummy, layer_num).A;
                    var costPlus = compute_cost(forwardPlus, Y, lambda, parametersDummy, layer_num);
                    parametersDummy[item].matrix[i][j] -= 1e-4;
                    var forwardMinus = L_model_forward(X, parametersDummy, layer_num).A;
                    var costMinus = compute_cost(forwardMinus, Y, lambda, parametersDummy, layer_num);
                    gradient_prob.push([(costPlus - costMinus) / 1e-4]);
                    //print(gradient_prob.length);
                }
            }
        }
        var gradient = new Array();
        for (var l = 1; l < layer_num; l++) {
            gradient = gradient.concat(grads[l]["dW"].toVector().matrix).concat(grads[l]["db"].toVector().matrix);
        }
        gradient_prob = new Matrix(gradient_prob);
        gradient = new Matrix(gradient);
        //gradient_prob.minus(gradient).show()
        var numerator = gradient_prob.minus(gradient).norm().matrix[0][0];
        var denominator = gradient_prob.add(gradient).norm().matrix[0][0];
        var differenceRatio = numerator / denominator;
        if (differenceRatio > 2e-4) {
            print("There is a mistake in the backward propagation! difference = " + differenceRatio);
        } else {
            print("Your backward propagation works perfectly fine! difference = " + differenceRatio);
        }
    }
    //return accuracy
    NN.prototype.fit = function (X, Y) {
        var predict = function (data_X, data_Y, parameters, layer_num) {
            var forwardResult = L_model_forward(data_X, parameters, layer_num);
            var correct = 0;
            var inCorrect = 0;
            for (var i = 0; i < forwardResult.A.shape()[1]; i++) {
                if (Math.abs(data_Y.matrix[0][i] - forwardResult.A.matrix[0][i]) < 0.5) {
                    correct++;
                } else {
                    inCorrect++;
                }
            }
            var accuracy = correct / (correct + inCorrect);
            print("The accuracy is " + accuracy * 100 + "%");
            return accuracy;
        };
        X = new Matrix(X);
        Y = new Matrix(Y);
        X = X.T();
        Y = Y.T();
        var parameters = initialize_parameters(this.layer_dims, this.init_method);
        for (var i = 1; i <= this.iterations; i++) {
            var forward = L_model_forward(X, parameters, this.layer_num, this.keep_prob);
            var gradients = L_model_backward(forward.A, Y, forward.cache, this.layer_num, this.lambda, this.keep_prob);
            if (this.gradient_check && !(this.keep_prob < 1000)) {
                gradient_check(parameters, gradients, this.layer_num, this.lambda, X, Y);
            }
            parameters = update_parameters(parameters, gradients, this.learning_rate, this.layer_num);
            if (this.print_cost && i % 100 === 0) {
                print("Cost after iteration " + i + ": " + compute_cost(forward.A, Y, this.lambda, parameters, this.layer_num));
                //cost.show();
            }
        }
        return predict(X, Y, parameters, this.layer_num);
    };
    ml.NN = NN;
})(ML);
