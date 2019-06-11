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
z = X*theta
hypot =  sigmoid(z)
% j_s = -y(iter)*log(hypot(iter)) - 1 - y(iter) * log(1-hypot(iter)) + lambda/2m * theta^2
J = ((1/m) * sum(-y'*log(hypot) - (1-y)'*log(1 - hypot))) + (lambda/(2*m))*sum(theta(2:end,1).^2)
% grad(1) --> first row of gradients for when theta(j) and j == 0. 
% X(1:end,1) --> all rows, first column of X 
grad(1)= (1/m)*X(1:end,1)'*(hypot-y)
% grad(2:end,1) --> all rows except first when theta(j) and j >= 1. 
% X(1:end,2:end) --> all rows, all columns except the 1st one 
grad(2:end, 1)= ((1/m)*X(1:end,2:end)'*(hypot-y)) + (lambda/m)*theta(2:end,1)




% =============================================================

end