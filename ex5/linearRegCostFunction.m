function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J = 0;
h = X*theta;
error = h - y;
error_square = error.^2;
result = sum(error_square);
J = result/(2*m);
% performing regularization
theta(1) = 0;
reg =  lambda/(2*m) * sum(theta.^2);
J = J + reg;
grad = (X' * error)/m + (lambda/m) * theta;
% =========================================================================

end
