function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

tryC = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
trySigma = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

% function to calculate prediction error for given `C` and `sigma`
calculateError = @(C, sigma) mean(double(
  svmPredict(
    svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)),
    Xval
  ) ~= yval
));

% call the `calculateError` function on all combinations of `C` and `sigma`
errors = arrayfun(
  calculateError,
  % the two matrices of the same dimensions:
  % - tryC: repeats all values of `C` to try on `numSigma` identical rows
  % - trySigma: repeats all values of `sigma` to try on `numC` identical columns
  % each positions (`i`,`j`) provides different (`C`, `sigma`) pair of constants
  repmat(tryC, length(trySigma), 1),
  repmat(trySigma', 1, length(tryC), 1)
);

% find the minimum prediction error and return associated hyperparameters
[minSigmaIdx, minCIdx] = find(errors == min(min(errors)));
C = tryC(minCIdx);
sigma = trySigma(minSigmaIdx);

% =========================================================================

end
