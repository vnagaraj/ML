function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

row_count = size(g, 1);
col_count = size(g, 2);
for row = 1:row_count,
	for col = 1:col_count,
		val = z(row, col);
		val = 1/(1 + exp(val * -1));
		g(row, col) = val;
	end;
end;
% =============================================================

end
