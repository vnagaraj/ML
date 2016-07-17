function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
theta = theta'; % transpose theta for matrix multiplication with x
m = size(y, 1); % training example count
n = size(theta, 2); % parameter count
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
result = result/m;
reg  = 0;
for i = 2:n,
	reg += theta(1,i)**2;
end;
reg = lambda/(2 * m) * reg;
J = result + reg;
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
	if j == 1, 
	  reg_factor = 0;
	else,
 	  reg_factor = lambda/m * theta(1,j);
	end;
	grad(j,1) = result/m + reg_factor;
end;
% =============================================================

end	
