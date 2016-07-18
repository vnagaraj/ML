function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    k = size(theta,1);
    temp = rand(k,1);
    scale = alpha/m;
    h_vector = X * theta;
    error_vector = h_vector - y;
    x_transpose = X';
    theta_change = (x_transpose * error_vector) * scale;
    temp = theta - theta_change;
    changed = 0;
    for j = 1:k,
       if theta(j,: ) ~= temp(j,: ),
          changed = 1;
       end;          
       theta(j,:) = temp(j,:);
    end 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    if changed == 0,
      break;
    end;

end

end
