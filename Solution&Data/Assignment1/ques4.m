table1 = xlsread('D:\CMS\Sem3-1\Neural Networkand Fuzzy Logic\data.xlsx');
% X is the matrix containing feature vectors for all instances
X = ones(size(table1,1),1);
X = [X table1(:,1:2)];
% Inputs normalization
X(:,2) = (X(:,2)-mean(X(:,2)))/std(X(:,2));
X(:,3) = (X(:,3)-mean(X(:,3)))/std(X(:,3));
%Target outputs
y = (table1(:,3)-mean(table1(:,3)))/std(table1(:,3));
clear table1
w = (X'*X)\X'*y; %Weight evaluation using vectorised linear regression
w_bgd=[1.13e-15;0.0781;0.3606]; %Weights from linear regression - batch gradient descent
w_sgd=[-0.0029;0.0990;0.2951];%Weights from linear regression - stochastic gradient descent
e1 = sqrt(sum((w-w_bgd).^2)); %Error with respect to batch gradient descent
e2 = sqrt(sum((w-w_sgd).^2)); %Error with respect to stochastic gradient descent