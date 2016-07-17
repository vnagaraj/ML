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
a1 = [ones(m, 1) X]; %adding bias unit to input
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; %adding bias unito to a1
z3 = a2 * Theta2';
a3 = sigmoid(z3); % this is the h function
log_hyp = log(a3);
y_matrix = eye(num_labels)(y,:);
minus_y = -1 * y_matrix;
first_term = (minus_y) .* log_hyp;
log_hyp_inv = log(1 - a3);
second_term = (1 - y_matrix) .* log_hyp_inv;
cost = first_term - second_term;
cost = sum(sum(cost))/m;
theta1_cpy = Theta1;
theta1_cpy(:,1)= []; % to compute regularization eliminate bias from first column
theta2_cpy = Theta2;
theta2_cpy(:,1)= []; % to compute regularization eliminate bias from first column
theta_vector = [ theta1_cpy(:); theta2_cpy(:)]; %unrolling theta1 and theta2 into single vector after eliminating bias
reg = lambda/(2*m) * (theta_vector' * theta_vector);
J = cost + reg;

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
delta3_matrix = ones(m, num_labels);
a3_matrix = ones(m, num_labels);
delta2_matrix = ones(m, hidden_layer_size);
a2_matrix = ones(m, hidden_layer_size+1);
a1_matrix = ones(m, input_layer_size+1);
for t = 1:m,
  % forward progagation to compute activation units for all layers
  a1 = X(t,:);
  a1 = [1;a1'];
  a1_matrix(t,:) = a1;
  z2 = a1' * Theta1';
  a2 = sigmoid(z2);
  a2 = [1;a2']; %adding bias unito to a1
  a2_matrix(t,:) = a2;
  z3 = a2' * Theta2';
  a3 = sigmoid(z3); % this is the h function
  a3_matrix(t,:) = a3;
  % backward propagation to compute gradients
  y = y_matrix(t,:)';
  delta3 = a3 - y';
  delta3_matrix(t,:) = delta3;
  delta2 = (Theta2' * delta3');
  delta2 = delta2(2:end);
  delta2 = delta2' .* (sigmoidGradient(z2));
  delta2_matrix(t,:) = delta2;
end;
Delta2_matrix = delta3_matrix' * a2_matrix;
Delta1_matrix = delta2_matrix' * a1_matrix;
Theta1_grad = [Delta1_matrix]/m;
Theta2_grad = [Delta2_matrix]/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% to account for regularization
if lambda != 0,
 for i = 1: size(Theta2_grad,1),
   for j = 2: size(Theta2_grad, 2),
	reg = Theta2(i,j) * lambda/m;
	Theta2_grad(i,j) = Theta2_grad(i,j) + reg;
   end;
 end;
 for i = 1: size(Theta1_grad,1),
   for j = 2: size(Theta1_grad, 2),
	reg = Theta1(i,j) * lambda/m;
	Theta1_grad(i,j) = Theta1_grad(i,j) + reg;
   end;
 end;
end;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
