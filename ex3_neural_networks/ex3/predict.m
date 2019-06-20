function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add ones to Matrix X 
a1 = [ones(m, 1) X]; % a1 will be 5000 * 401
% get z2 by multiplying a1 by its weights. 
z2 = a1 * Theta1'; % Theta1' is 401 * 25 
% Calculate the 
a2_prime = sigmoid(z2) % a2 will be 5000 * 25 

% Calculate output layer 
% add ones to matrix of a2 
m2 = size(a2_prime, 1); 
a2 = [ones(m2, 1) a2_prime] % a2 will be 5000 * 26 
z3 = a2 * Theta2' % Theta2' will be 26 * 10 

% calculate a3
a3 = sigmoid(z3) % a3 will be 5000 * 10
[preds_max, idx_max] = max(a3, [], 2)

p = idx_max % p predictions will be a vector of 5000 * 1 






% =========================================================================


end
