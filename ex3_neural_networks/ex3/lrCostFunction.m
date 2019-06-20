function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

z = X*theta
hypot =  sigmoid(z)
% j_s = -y(iter)*log(hypot(iter)) - 1 - y(iter) * log(1-hypot(iter)) + lambda/2m * theta^2
% calculate the cost 
J = ((1/m) * sum(-y'*log(hypot) - (1-y)'*log(1 - hypot))) + (lambda/(2*m))*sum(theta(2:end,1).^2)

% Calculate the Gradients, remembering that we don't add for j = 0 a
% grad(1) --> first row of gradients for when theta(j) and j == 0. 
% X(1:end,1) --> all rows, first column of X 
grad(1)= (1/m)*X(1:end,1)'*(hypot-y)
% grad(2:end,1) --> all rows except first when theta(j) and j >= 1. 
% X(1:end,2:end) --> all rows, all columns except the 1st one 
grad(2:end, 1)= ((1/m)*X(1:end,2:end)'*(hypot-y)) + (lambda/m)*theta(2:end,1)










% =============================================================

grad = grad(:);

end
