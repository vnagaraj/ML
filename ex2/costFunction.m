function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
theta = theta'; % transpose theta for matrix multiplication with x
m = size(y, 1); % training example count
grad_count = size(grad, 1); % parameter theta count
result = 0;
for i = 1:m,
	x_i = X(i, :)';
	hyp = sigmoid(theta * x_i); % matrix multiplication
	y_i = y(i, :);
	first_term = (-1 * y_i) * log(hyp);
	second_term = (1 - y_i) * log(1-hyp);
	result += first_term - second_term;
end;
J = result/m;
for j = 1:grad_count,	
	result = 0;
	for i = 1:m,
		x_i = X(i, :)';
		hyp = sigmoid(theta * x_i); % matrix multiplication
		y_i = y(i, :);
		first_term = hyp - y_i;
		x_j_i = x_i(j,:);
		result += first_term * x_j_i;
	end;
	grad(j,1) = result/m;	
end;


	






% =============================================================

end
