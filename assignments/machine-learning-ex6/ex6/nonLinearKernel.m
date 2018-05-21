function sim = nonLinearKernel(x1, x2)
%NONLINEARKERNEL returns a non-linear kernel between x1 and x2
%   sim = linearKernel(x1, x2) returns a non-linear kernel between x1 and x2
%   and returns the value in sim
%
%
%   This is a custom kernel created for the example of 4-points
%   separable by hyperbola:
%
%   X = [ 0, 3; 1, 2; 2, 1; 3, 0 ]
%   y = [ 0; 1; 1; 0 ]
%
%   transformed to (separable by a plane in 3D space)
%
%   X = [ 0, 3, 0; 1, 2, 1; 2, 1, 1; 3, 0, 0 ]
%   y = [ 0; 1; 1; 0 ]

% Add transformation
x1 = [ x1(:); prod(x1) ]; x2 = [ x2(:); prod(x2) ];

% Compute the kernel
sim = x1' * x2;  % dot product

end
