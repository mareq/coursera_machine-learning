function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% - eye(num_labels) creates identity matrix
% - (y, :) select appropriate row (and with all its columns) from that
%   identity matrix
% - this is repeated for each value in `y` and the results are stacked
%   as rows into the resulting `size(y,1)`x`num_labels` matrix (5000x10)
Y = eye(num_labels)(y, :);
% Thetas without bias (cut off the first column)
Theta1WoBias = Theta1(:, 2:end);
Theta2WoBias = Theta2(:, 2:end);

% ============================================================================
% Part 1: Forward Propagation and Cost Computation
% ============================================================================

% Input Layer
X
a1 = [ones(m, 1), X];

% Hidden Layer
z2 = a1 * Theta1';
a2 = sigmoid(z2);
bias = ones(size(a2, 1), 1);
a2 = [bias, a2];

% Output Layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Cost Computation
H = a3;
% - both `H` and `Y` matrices have the same dimensions
% - cost computation is done for each sample & class
% - resulting costs are summed up (sum over samples, then sum over classes)
J = (1 / m) * sum(sum((-Y .* log(H)) - ((1-Y) .* log(1-H))));

% Regularization
%r = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
% - unroll Theta matrices into single columnar vector
ThetasWoBias = [Theta1WoBias(:); Theta2WoBias(:)];
% - run along the long horizontal vector and sum the squares ov the values
r = (lambda / (2 * m)) * sum(ThetasWoBias' .^ 2);
J += r;


% ============================================================================
% Part 2: Back Propagation
% ============================================================================

% Step 1: Forward Propagation
% - already done in Part 1:
% - activations for t-th sample: aN(t, :), zN(t, :)

% Step 2: Delta for the Output Layer
% - delta for t-th sample and k-th output unit: d3(t,k)
d3 = a3 - Y;

% Step 3: Delta for the Hidden Layer
% - delta for t-th sample and k-th unit on hidden layer: d2(t,k)
d2 = (d3 * Theta2WoBias) .* sigmoidGradient(z2);

% Step 4: Accumulate Gradient
D2 = d3' * a2;
D1 = d2' * a1;

% Step 5: Unregularized Gradient
Theta1_grad = (1 / m) .* D1;
Theta2_grad = (1 / m) .* D2;

% ============================================================================
% Part 2: Regularization
% ============================================================================

Theta1_grad(:, 2:end) += (lambda / m) * Theta1WoBias;
Theta2_grad(:, 2:end) += (lambda / m) * Theta2WoBias;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
