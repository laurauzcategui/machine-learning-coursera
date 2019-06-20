function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% Calculate the activation function for X with the gradients
% it will return a matrix of 5000 * 10, each row 
% representing an observation, and 10 columns each with the probability to be
% that number. 
% Example: preds_t(3,:)
% Output: 
% Columns 1 through 4:
% 0.00000000281623   0.00000002023655   0.00080606353807   0.00000000021822 
% Columns 5 through 8:  
% 0.00896944168435   0.00029888042176   0.00000104394035   0.03341882102029
% Columns 9 and 10:
% 0.00017674361977   0.58360892804194

preds = sigmoid(X * all_theta');
% Now we are interested to take the for each observation and return the 
% maximum probability of being number Z, we use  max function and 
% store the actual maximum probability and the index of that probability 
[max_prob, idx_max_prob] = max(preds, [],2);
% Since we are interested only to know the number we predicted 
% we stored in p ( vector of 5000x1), and we store the index which will 
% match a label for that prediction for each observation
p = idx_max_prob;






% =========================================================================


end
