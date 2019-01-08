//Matrix opertation libaray for JS

var NN = NN || {};
(function (ml) {
    var init = function (config) {
        config = config || {};
        if (!config.init_method) {
            //default: random inintialization
            print("Initialization method is set as default (random inintialization).");
            config.init_method = null;
        }
        if (!config.mini_batch_size) {
            print("Mini-batch size is set as default (the number of the training sets).");
            config.mini_batch_size = null;
        }
        if (!config.layer_dims) {
            throw "Error: please specify the NN layer structure in an array";
        }
        if (!config.opti_method) {
            print("Optimization method is set as default (gradient descent).");
            config.opti_method = null;
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
        if (!config.beta1) {
            config.beta1 = 0.9;
        }
        if (!config.beta2) {
            config.beta2 = 0.999;
        }
        if (!config.keep_prob) {
            config.keep_prob = 1;
        }
        if (!config.print_cost) {
            config.print_cost = false;
        }
        if (!config.gradient_check) {
            config.gradient_check = false;
        }
        this.mini_batch_size = config.mini_batch_size;
        this.layer_dims = config.layer_dims;
        this.iterations = config.iterations;
        this.learning_rate = new Matrix([[config.learning_rate]]);
        this.lambda = new Matrix([[config.lambda]]);
        this.beta1 = config.beta1;
        this.beta2 = config.beta2;
        this.keep_prob = config.keep_prob;
        this.init_method = config.init_method;
        this.opti_method = config.opti_method;
        this.print_cost = config.print_cost;
        this.gradient_check = config.gradient_check;
        this.layer_num = config.layer_dims.length - 1;
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
    //matrix library for operation
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
        this.shuffle = function (matrix) {
            var length1 = this.matrix.length;
            var width1 = this.matrix[0].length;
            var length2 = matrix.matrix.length;
            for (var j = width1 - 1; j > 0; j--) {
                var random_index = Math.floor(Math.random() * (j + 1));
                for (var i = 0; i < length1; i++) {
                    var temp1 = this.matrix[i][j];
                    this.matrix[i][j] = this.matrix[i][random_index];
                    this.matrix[i][random_index] = temp1;
                }
                for (var i = 0; i < length2; i++) {
                    var temp2 = matrix.matrix[i][j];
                    matrix.matrix[i][j] = matrix.matrix[i][random_index];
                    matrix.matrix[i][random_index] = temp2;
                }
            }
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
    //return .X and .Y
    init.prototype.random_mini_batches = function (X, Y) {
        var X_mini = new Array();
        var Y_mini = new Array();
        var X_train = new Array();
        var Y_train = new Array();
        var m = X.shape()[1];
        if (!this.mini_batch_size) {
            this.mini_batch_size = m;
        }
        var num = Math.floor(m / this.mini_batch_size);
        X.shuffle(Y);
        for (var k = 0; k < num; k++) {
            for (var i = 0; i < X.shape()[0]; i++) {
                X_mini[i] = X.matrix[i].slice(k * this.mini_batch_size, (k + 1) * this.mini_batch_size - 1);
            }
            X_train.push(new Matrix(X_mini));
        }
        for (var k = 0; k < num; k++) {
            for (var i = 0; i < Y.shape()[0]; i++) {
                Y_mini[i] = Y.matrix[i].slice(k * this.mini_batch_size, (k + 1) * this.mini_batch_size - 1);
            }
            Y_train.push(new Matrix(Y_mini));
        }
        if (m % this.mini_batch_size !== 0) {
            for (var i = 0; i < X.shape()[0]; i++) {
                X_mini[i] = X.matrix[i].slice(num * this.mini_batch_size, m - 1);
            }
            X_train.push(new Matrix(X_mini));
            for (var i = 0; i < Y.shape()[0]; i++) {
                Y_mini[i] = Y.matrix[i].slice(num * this.mini_batch_size, m - 1);
            }
            Y_train.push(new Matrix(Y_mini));
        }
        return {
            X: X_train,
            Y: Y_train
        };
    };
    //return v[No.layer]["W"/"b"]
    init.prototype.initialize_velocity = function () {
        var v = new Array();
        var L = this.layer_dims.length;
        for (var l = 1; l < L; l++) {
            v[l] = new Object();
            v[l]["dW"] = new Matrix();
            v[l]["db"] = new Matrix();
            v[l]["dW"].create([this.layer_dims[l], this.layer_dims[l - 1]], 0);
            v[l]["db"].create([this.layer_dims[l], 1], 0);
        }
        return v;
    };
    //return v[No.layer]["W"/"b"]
    init.prototype.initialize_adam = function () {
        var v = new Array();
        var s = new Array();
        var L = this.layer_dims.length;
        for (var l = 1; l < L; l++) {
            v[l] = new Object();
            s[l] = new Object();
            v[l]["dW"] = new Matrix();
            v[l]["db"] = new Matrix();
            s[l]["dW"] = new Matrix();
            s[l]["db"] = new Matrix();
            v[l]["dW"].create([this.layer_dims[l], this.layer_dims[l - 1]], 0);
            v[l]["db"].create([this.layer_dims[l], 1], 0);
            s[l]["dW"].create([this.layer_dims[l], this.layer_dims[l - 1]], 0);
            s[l]["db"].create([this.layer_dims[l], 1], 0);
        }
        return {
            v: v,
            s: s};
    };
    //return cache["W"/"b" + No.layer]
    init.prototype.initialize_parameters = function () {
        var parameters = new Object();
        var L = this.layer_dims.length;
        for (var l = 1; l < L; l++) {
            parameters["W" + String(l)] = new Matrix();
            parameters["b" + String(l)] = new Matrix();
            parameters["W" + String(l)].create([this.layer_dims[l], this.layer_dims[l - 1]]);
            parameters["b" + String(l)].create([this.layer_dims[l], 1]);
            if (this.init_method === "He") {
                parameters["W" + String(l)].mul([[Math.pow(2 / this.layer_dims[l - 1], 0.5)]]);
            }
        }
        return parameters;
    };
    //return A, cache["W"/"b"/"Z"/"A"/"D"]
    init.prototype.linear_activation_forward = function (A_prev, W, b, activation) {
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
        var linearForward = linear_forward(A_prev, W, b, this.keep_prob);
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
    };
    //return A, cache[No.layer]
    init.prototype.L_model_forward = function (X, parameters) {
        var cache = new Array();
        var A_prev = X;
        for (var l = 1; l < this.layer_num; l++) {
            var forward = this.linear_activation_forward(A_prev, parameters["W" + String(l)], parameters["b" + String(l)], "relu");
            A_prev = forward.A;
            cache[l] = forward.cache;
        }
        forward = this.linear_activation_forward(A_prev, parameters["W" + String(this.layer_num)], parameters["b" + String(this.layer_num)], "sigmoid");
        cache[this.layer_num] = forward.cache;
        return{A: forward.A,
            cache: cache
        };
    };
    //return cost value
    init.prototype.compute_cost = function (X, Y, parameters) {
        var m = Y.shape()[1];
        var one = new Matrix([[1]]);
        var AL = this.L_model_forward(X, parameters).A;
        if (AL.shape()[0] !== Y.shape()[0]) {
            throw "Error: unable to compute the cost with the dimension of the input matrixes of (" + AL.shape()[0] + "," + AL.shape()[1] + ") and (" + Y.shape()[0] + "," + Y.shape()[1] + ")";
        }
        //AL.show();
        //AL.log().show();
        var cost = (Y.mul(AL.log()).add(one.minus(Y).mul(one.minus(AL).log()))).sum().mul([[-1 / m]]);
        if (this.lambda > 0) {
            var sum = 0;
            for (var l = 1; l <= this.layer_num; l++) {
                sum += parameters["W" + String(l)].pow(2).sum().matrix[0][0];
            }
            var L2_cost = this.lambda.div([[2 * m]]).mul([[sum]]);
            cost = cost.add(L2_cost);
        }
        return cost.matrix[0][0];
    };
    //return dW, db, dA_prev
    init.prototype.linear_activation_backward = function (dA, cache, activation) {
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
        var grad = linear_backward(dZ, cache, this.lambda, this.keep_prob);
        return grad;
    };
    //return grad[No.layer]
    init.prototype.L_model_backward = function (AL, Y, cache) {
        var Y = new Matrix(Y);
        var grad = new Array();
        var dAL = Y.minus([[1]]).div(AL.minus([[1]])).minus(Y.div(AL));
        grad[this.layer_num] = this.linear_activation_backward(dAL, cache[this.layer_num], "sigmoid");
        for (var l = this.layer_num - 1; l > 0; l--) {
            grad[l] = this.linear_activation_backward(grad[l + 1]["dA"], cache[l], "relu");
        }
        return grad;
    };
    //return updateParameters["W"/"b" + No.layer]
    init.prototype.update_parameters = function (parameters, grads) {
        var updateParameters = new Object();
        for (var l = 1; l <= this.layer_num; l++) {
            updateParameters["W" + String(l)] = parameters["W" + String(l)].minus(this.learning_rate.mul(grads[l]["dW"]));
            updateParameters["b" + String(l)] = parameters["b" + String(l)].minus(this.learning_rate.mul(grads[l]["db"]));
        }
        return updateParameters;
    };
    init.prototype.update_parameters_momentum = function (parameters, grads, v) {
        var updateParameters = new Object();
        for (var l = 1; l <= this.layer_num; l++) {
            //v update
            v[l]["dW"] = v[l]["dW"].mul([[this.beta1]]).add(grads[l]['dW'].mul([[1 - this.beta1]]));
            v[l]["db"] = v[l]["db"].mul([[this.beta1]]).add(grads[l]['db'].mul([[1 - this.beta1]]));
            //parameter["W"/"b"] update
            updateParameters["W" + String(l)] = parameters["W" + String(l)].minus(this.learning_rate.mul(v[l]["dW"]));
            updateParameters["b" + String(l)] = parameters["b" + String(l)].minus(this.learning_rate.mul(v[l]["db"]));
        }

        return updateParameters;
    };
    init.prototype.update_parameters_adam = function (parameters, grads, v, s, t) {
        var updateParameters = new Object();
        var v_correct = new Array();
        var s_correct = new Array();
        for (var l = 1; l <= this.layer_num; l++) {
            //v update
            v_correct[l] = new Object();
            v[l]["dW"] = v[l]["dW"].mul([[this.beta1]]).add(grads[l]['dW'].mul([[1 - this.beta1]]));
            v_correct[l]["dW"] = v[l]["dW"].div([[1 - Math.pow(this.beta1, t)]]);
            v[l]["db"] = v[l]["db"].mul([[this.beta1]]).add(grads[l]['db'].mul([[1 - this.beta1]]));
            v_correct[l]["db"] = v[l]["db"].div([[1 - Math.pow(this.beta1, t)]]);
            //s update
            s_correct[l] = new Object();
            s[l]["dW"] = s[l]["dW"].mul([[this.beta2]]).add(grads[l]['dW'].pow(2).mul([[1 - this.beta2]]));
            s_correct[l]["dW"] = s[l]["dW"].div([[1 - Math.pow(this.beta2, t)]]);
            s[l]["db"] = s[l]["db"].mul([[this.beta2]]).add(grads[l]['db'].pow(2).mul([[1 - this.beta2]]));
            s_correct[l]["db"] = s[l]["db"].div([[1 - Math.pow(this.beta2, t)]]);
            //parameter["W"/"b"] update
            updateParameters["W" + String(l)] = parameters["W" + String(l)].minus(this.learning_rate.mul(v_correct[l]["dW"]).div(s_correct[l]["dW"].pow(0.5).add([[1e-7]])));
            updateParameters["b" + String(l)] = parameters["b" + String(l)].minus(this.learning_rate.mul(v_correct[l]["db"]).div(s_correct[l]["db"].pow(0.5).add([[1e-7]])));
            //updateParameters["W" + String(l)].minus(parameters["W" + String(l)]).show()
        }
        return updateParameters;
    };
    init.prototype.gradient_check = function (parameters, grads, X, Y) {
        var gradient_prob = new Array();
        for (var item in parameters) {
            var length = parameters[item].shape()[0];
            var width = parameters[item].shape()[1];
            //print(item)
            for (var i = 0; i < length; i++) {
                for (var j = 0; j < width; j++) {
                    var parametersDummy = parameters;
                    parametersDummy[item].matrix[i][j] += 1e-4;
                    var forwardPlus = this.L_model_forward(X, parametersDummy, this.layer_num).A;
                    var costPlus = this.compute_cost(forwardPlus, Y, this.lambda, parametersDummy, this.layer_num);
                    parametersDummy[item].matrix[i][j] -= 1e-4;
                    var forwardMinus = this.L_model_forward(X, parametersDummy, this.layer_num).A;
                    var costMinus = this.compute_cost(forwardMinus, Y, this.lambda, parametersDummy, this.layer_num);
                    gradient_prob.push([(costPlus - costMinus) / 1e-4]);
                    //print(gradient_prob.length);
                }
            }
        }
        var gradient = new Array();
        for (var l = 1; l <= this.layer_num; l++) {
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
    init.prototype.predict = function (data_X, data_Y, parameters) {
        var forwardResult = this.L_model_forward(data_X, parameters);
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
    //return accuracy
    init.prototype.train = function (X, Y) {
        //training set format must be: (training-set size, training-set number)
        //format the training sets
        X = new Matrix(X).T();
        Y = new Matrix(Y).T();
        print("=====Training Started=====");
        var parameters = this.initialize_parameters();
        if (this.opti_method === "momentum") {
            var v = this.initialize_velocity();
        } else if (this.opti_method === "adam") {
            var init = this.initialize_adam();
            var v = init.v;
            var s = init.s;
            var t = 0;
        }
        for (var iter = 1; iter <= this.iterations; iter++) {
            var trainingSet = this.random_mini_batches(X, Y);
            for (var i = 0; i < trainingSet.X.length; i++) {
                var forward = this.L_model_forward(trainingSet.X[i], parameters);
                var gradients = this.L_model_backward(forward.A, trainingSet.Y[i], forward.cache);
                if (this.gradient_check && !(this.keep_prob < 1000)) {
                    gradient_check(parameters, gradients, this.layer_num, this.lambda, X, Y);
                }
                if (this.opti_method === "momentum") {
                    parameters = this.update_parameters_momentum(parameters, gradients, v);
                } else if (this.opti_method === "adam") {
                    t = t + 1;
                    parameters = this.update_parameters_adam(parameters, gradients, v, s, t);
                } else {
                    parameters = this.update_parameters(parameters, gradients);
                }
            }
            if (this.print_cost && iter % 1 === 0) {
                print("Cost after iteration " + iter + ": " + this.compute_cost(X, Y, parameters));
                //cost.show();
            }
        }
        print("=====Training Finished=====");
        return this.predict(X, Y, parameters);
    };
    ml.init = init;
})(NN);
