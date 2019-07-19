function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypot = X*theta;
% 2*m 
m_by_two = 2*m;

% Cost function implementation
% Excluding theta_0 from the regularization
J = (1/m_by_two) * sum((hypot - y).^2) + ((lambda/m_by_two)* sum(theta(2:end, 1).^2));

% Calculating the gradient 
grad(1) =  (1/m) * (X(1:end, 1)'*(hypot-y));
grad(2:end) = ((1/m) * (X(1:end, 2:end)'*(hypot-y))) + (lambda/m)*theta(2:end, 1);









% =========================================================================

grad = grad(:);

end
