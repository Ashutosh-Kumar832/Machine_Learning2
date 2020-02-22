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
g = zeros(size(theta));
n = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
h = sigmoid(z);

q = -y.*log(h);
r = (1-y).*log(1-h);
s = theta(2:length(theta)).^2;
a = sum(q - r)/m;
b = (sum(s)*lambda)/(2*m);
J = a + b;

for i = 1 : length(theta)
    if i>1
        g(i) = sum((h - y).*X(:,i))/m;
        n(i) = (lambda/m)*theta(i);
    else
        g(i) = sum((h - y).*X(:,i))/m;
    end
end
grad = g+n;
% =============================================================
end
