function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta = theta';
    result = 0;
    k = size(theta,2);
    temp = rand(1,k);
    for j = 1:k,
        for i = 1:m,
          x = X(i,:)';
          x_index = x(j,:);
          result += ((theta * x) - y(i)) * x_index;
        end;
        result=  alpha * result/m;
        temp(:,j) = theta(:,j) - result;
    end;
    x = 0;
    for j = 1:k,
       if theta(:, j) ~= temp(:, j),
          x = 1;
       end;          
       theta(:, j) = temp(:,j);
    end 
    theta = theta';
    disp(theta)
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta)
    

end

end
