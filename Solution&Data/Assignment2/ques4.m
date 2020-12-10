clc; clear all;
X = load('D:\CMS\Sem3-1\Neural Networkand Fuzzy Logic\data5.mat');
X=X.x;
X(:,1:72) = (X(:,1:72)-mean(X(:,1:72)))./std(X(:,1:72));
p = randperm(size(X,1));
X1 = ones(2148,73);
for i=1:size(X,1)
    X1(i,:) = X(p(i),:);
    if(X1(i,73)==0)
        X1(i,73)=-1;
    end
end
IA = [0 0];
W = zeros(10000,500); %number of nodes in hidden layer
Xtrain = X1(1:2000,1:72);
Ytrain = X1(1:2000,73);
Xtest = X1(2001:2148,1:72);
Ytest = X1(2001:2148,73);
randommat = randn(73,1000);
Xtrain = [ones(size(Xtrain,1),1) Xtrain];
G = Xtrain*randommat;
H = tanh(G);
W = pinv(H)*Ytrain;
count=0;
for j=1:148
    b(j) = testELM(Xtest(j,:),randommat,W);
    if b(j)>0
        b(j)=1;
        IA(1)=IA(1)+(Ytest(j)==1);
    else
        b(j)=-1;
        IA(2) = IA(2)+(Ytest(j)==-1);
    end
    if(b(j)==Ytest(j))
        count=count+1;
    end
end
IA(1) = IA(1)/sum(Ytest(:)==1);
IA(2) = IA(2)/sum(Ytest(:)==-1);
Accuracy = count/148;
%function%
function y = testELM(features,randomperm,w)
    a = [1 features];
    g = a*randomperm;
    h = cos(g);
    y = h*w;
end