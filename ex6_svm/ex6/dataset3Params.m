function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% start by setting random values to the parameters
% until you see the error_pred stop decreasing
C = 0.95;
sigma = 0.07;

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
% change this parameters accordingly
% if you wish to see a different rate of change
max_iter = 5; 
step = 0.01;

min_pred_error = 1; 
min_c = C;
min_sigma = sigma; 
for iter = 1:max_iter 
  C += step; 
  sigma += step; 
  fprintf("Iter: %d New sigma = %f C = %d\n" ,iter,  sigma, C);
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

  predictions = svmPredict(model, Xval);
  pred_error = mean(double(predictions ~= yval));
  if pred_error < min_pred_error
    fprintf("New Pred_error=%f sigma = %f C = %d\n" , pred_error, sigma, C)
    min_pred_error = pred_error;
    if min_c < C 
      min_c = C;
    endif
    if min_sigma < sigma
      min_sigma = sigma;
    endif
  endif
endfor
visualizeBoundary(X, y, model);
fprintf('Pred_error= %f , sigma = %f C = %d\n' , min_pred_error, min_sigma, min_c);

C = min_c; 
sigma = min_sigma



% =========================================================================

end
