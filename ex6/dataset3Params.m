function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_vector= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vector = c_vector(:)';
min_error = 1000; % some high value
no_vals = size(c_vector,2);
for i = 1:no_vals,
 for j = 1:no_vals,
 current_c = c_vector(:,i);
 current_sigma = sigma_vector(:,j);
 model= svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma)); 
 predictions = svmPredict(model, Xval);
 error = mean(double(predictions ~= yval));
 if min_error > error,
   min_error = error;
   C = current_c;
   sigma = current_sigma;
 end;
 end;
end;

% =========================================================================

end
