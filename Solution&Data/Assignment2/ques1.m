clc;
close all;
A = load('D:\CMS\Sem3-1\Neural Networkand Fuzzy Logic\data5.mat');
A=A.x;
data=A;
%Input normalization
data(:,1:end-1) = (data(:,1:end-1)-mean(data(:,1:end-1)))./std(data(:,1:end-1));
X = [ones(size(data,1),1) data(:,1:end-1)]; %Inputs
Y = data(:,end); %Target outputs
MSETarget = 0.01; %Target MSE
H = 6; %Number of neurons in hidden layer 1
Q = 6; %Number of neurons in hidden layer 2
w = rand(size(X,2),H)/10; %Weights between input layer and hidden layer 1
v = rand(H+1,Q)/10; %Weights between hidden layer 1 and hidden layer 2
u = rand(Q+1,3)/10; %Weights between hidden layer 2 and output layer
lr = 0.01; %learning rate
%Converting target outputs to 3 output neurons
for i=1:length(Y)
    if Y(i)==1
        yt(i,:) = [1 0 0];
    else
        yt(i,:) = [0 0 1];
    end
end
%Randomly divide the dataset into training (70%) and testing (30%) set
p = randperm(length(Y));
trainInput = X(p(1:ceil(0.7*size(X,1))),:); trainOutput = yt(p(1:ceil(0.7*size(X,1))),:);
testInput = X(p(ceil(0.7*size(X,1))+1:end),:); testOutput = yt(p(ceil(0.7*size(X,1))+1:end),:);
it = 1;
er=1000;
while er>MSETarget %Stopping condition
    er = 0;
    p = randperm(length(trainOutput)); %Shuffling data before each iteration
    for t=1:length(trainOutput)
        z1 = logsig(w'*trainInput(p(t),:)'); %Output of hidden layer 1
        z1 = [1;z1]; %Concatenating input with 1 for bias
        z2 = logsig(v'*z1); %Output of hidden layer 2
        z2 = [1;z2]; %Concatenating hidden layer 1 output with 1 for bias
        y = logsig(u'*z2); %Output of output layer after sigmoid activation
        del = (trainOutput(p(t),:)'-y).*y.*(1-y);
        for q=1:Q+1
            delu(q,:) = -lr*del'*z2(q); %Change in u
        end
        z2 = z2(2:end);
        delv = zeros(H+1,Q);    %Change in v
        for h=1:H+1
            for q=1:Q
                for k=1:3
                    delv(h,q) = delv(h,q)-lr*del(k)*u(q+1,k)*z2(q)*(1-z2(q))*z1(h);
                end
            end
        end
        delw = zeros(size(X,2),H); %Change in w
        z1 = z1(2:end);
        for j=1:size(trainInput,2)
            for h=1:H
                for q=1:Q
                    for k=1:3
                        delw(j,h) = delw(j,h)-lr*del(k)*u(q+1,k)*z2(q)*(1-z2(q))*v(h+1,q)*z1(h)*(1-z1(h))*trainInput(p(t),(j));
                    end
                end
            end
        end
        %Weight updates
        w = w-delw;
        v = v-delv;
        u = u-delu;
        er = er+sum((trainOutput(p(t),:)'-y).^2); %Error calculation
    end
    er = er/(3*length(trainOutput)); %Cost calculation
    cost(it,1) = er;
    it = it+1;
end
plot(cost) %Plotting cost
xlabel('Number of iterations');
ylabel('Cost function');
title('Cost vs Number of iterations');
% Testing using testInput
for i=1:length(testOutput)
    z1 = logsig(w'*testInput(i,:)'); %Hidden layer 1 output
    z1 = [1;z1]; %Concatenating with 1 for bias
    z2 = logsig(v'*z1); %Hidden layer 2 output
    z2 = [1;z2]; %Concatenating with 1 for bias
    y = logsig(u'*z2); %Predicted output
    [~,yp(i)] = max(y); %Determining the class of predicted output
end
[~,ya] = max(testOutput,[],2); %Determining the class of test output
[cm, ~] = confusionmat(yp,ya); %Calculating confusion matrix
IA = zeros(1,2);
OA = 0;
for i = 1:2
    IA(i) = cm(i,i)/sum(cm(i,:)); %individual accuracy
    OA = OA + cm(i,i);
end
OA = OA/sum(cm(:)) %overall accuracy