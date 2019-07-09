function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Apply Feed Forward first
% add ones to Matrix X 
a1 = [ones(m, 1) X]; % a1 will be 5000 * 401
% get z2 by multiplying a1 by its weights. 
z2 = a1 * Theta1'; % Theta1' is 401 * 25 
% Calculate the 
a2_prime = sigmoid(z2); % a2 will be 5000 * 25 

% Calculate output layer 
% add ones to matrix of a2 
m2 = size(a2_prime, 1); 
a2 = [ones(m2, 1) a2_prime];% a2 will be 5000 * 26 
z3 = a2 * Theta2'; % Theta2' will be 26 * 10 

% calculate a3 or same as Ho(x)
a3 = sigmoid(z3); % a3 will be 5000 * 10 % a3 is the same as h0(x)

% we have an identity matrix with 1 on each position
I = eye(num_labels);
% we have a matrix of m*num_labels, i.e, 5000*10
Y = zeros(m, num_labels);

% We fill our new matrix with zeros and only 1 with the position of the right 
% number or label, 
% for example: y(2000) = 3, I(y(2000), :) = 0   0   1   0   0   0   0   0   0   0
for i = 1:m 
  Y(i, :) = I(y(i), :);
endfor

% calculate the cost function
J = 1/m * sum(sum((-Y.*log(a3)) - ((1-Y).*log(1-a3))));

% regularization term 
% Remember to index Theta 1 and Theta 2 as you don't want to 
% regularize the Bias parameters 
reg = (lambda / (2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% J regularized 
J = J + reg;

% calculate deltas
delta_3 = a3 .- Y;
size_z2 = size(z2,1);

delta_2 = (delta_3*Theta2) .* sigmoidGradient([ones(size_z2,1) z2]);
% remove the first column
delta_2 = delta_2(:, 2:end);
% Calculate then Delta
Theta1_grad = Theta1_grad + (delta_2' * a1);
Theta2_grad = Theta2_grad + (delta_3' * a2);

%without regularisation it will be 
% Theta1_grad = Theta1_grad / m;
% Theta2_grad = Theta2_grad / m;

% Let's regularise the gradients  when j == 0 
Theta1_grad(:,1) = Theta1_grad(:,1) / m;
Theta2_grad(:,1) = Theta2_grad(:,1) / m;

% Regularise with j >= 1 
Theta1_grad(:,2:end) = (Theta1_grad(:,2:end) / m) + ((lambda/m)*Theta1(:,2:end));
Theta2_grad(:,2:end) = (Theta2_grad(:,2:end) / m) + ((lambda/m)*Theta2(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
