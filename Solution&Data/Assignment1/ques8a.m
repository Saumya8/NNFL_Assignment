clc;
clear;
close all;
X = xlsread('D:\CMS\Sem3-1\Neural Networkand Fuzzy Logic\data4.xlsx');
%Input normalization
X(:,1:7) = (X(:,1:7)-mean(X(:,1:7)))./std(X(:,1:7));
%Shuffling data
p = randperm(size(X,1));
%Dividing the data into training data (60%) and testing data (40%)
train_data = [zeros(0.6*size(X,1),1) X(p(1:0.6*size(X,1)),:)];
test_data = [zeros(0.4*size(X,1),1) X(p(0.6*size(X,1)+1:end),:)];
clear p X
k = 500; %Number of iterations
alpha = 0.0001; %Learning rate
%Training 1 vs 2 and 3
t = train_data;
for i = 1:size(train_data,1)
    if train_data(i,9)==1
        t(i,9)=0;
    else
        t(i,9)=1;
    end
end
w1 = rand(1,size(t,2)-1); %initial weights for 1 vs 2 and 3
y = t(:,9); %target outputs
for i=1:k
    g = logsig(w1*(t(:,1:8)')); %hypothesis calculation
    %weight updates
    for j = 1:size(t,2)-1
        w1(j) = w1(j)-alpha*sum((y.*(1-g)+(y-1).*g)*t(:,j));
    end
end
clear y g
%Training 2 vs 1 and 3
t = train_data;
for i = 1:size(train_data,1)
    if train_data(i,9)==2
        t(i,9)=0;
    else
        t(i,9)=1;
    end
end
w2 = rand(1,size(t,2)-1); %initial weights for 2 vs 1 and 3
y = t(:,6); %target outputs
for i=1:k
    g = logsig(w2*t(:,1:8)')'; %hypothesis calculation
    %weight updates
    for j = 1:size(t,2)-1
        w2(j) = w2(j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
    end
end
clear y g
%Training 3 vs 1 and 2
t = train_data;
for i = 1:size(train_data,1)
    if train_data(i,9)==3
        t(i,9)=0;
    else
        t(i,9)=1;
    end
end
w3 = rand(1,size(t,2)-1); %initial weights for 3 vs 1 and 2
y = t(:,6); %target outputs
for i=1:k
    g = logsig(w3*t(:,1:8)')'; %hypothesis calculation
    %weight updates
    for j = 1:size(t,2)-1
        w3(j) = w3(j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
    end
end
clear y g t i j
%Testing with test data
yp_1 = logsig(w1*test_data(:,1:8)')'; %prediction of 1 vs 2 and 3
yp_2 = logsig(w2*test_data(:,1:8)')'; %prediction of 2 vs 1 and 3
yp_3 = logsig(w3*test_data(:,1:8)')'; %prediction of 3 vs 1 and 2
a = [yp_1 yp_2 yp_3];
yp = zeros(size(yp_1,1),1);
for i = 1:size(yp_1,1)
    %class is determined by the maximum prediction
    [~, yp(i)] = max(a(i,:));
end
c = zeros(3); %to calculate confusion matrix
for i = 1:size(yp,1)
    if test_data(i,9) == 1
        c(1,yp(i)) = c(1,yp(i))+1;
    elseif test_data(i,9) == 2
        c(2,yp(i)) = c(2,yp(i))+1;
    else
        c(3,yp(i)) = c(3,yp(i))+1;
    end
end
IA = zeros(1,3); %individual accuracies of the 3 predictors
Overall = 0; %overall accuracy
for i = 1:3
    IA(i) = c(i,i)/sum(c(i,:));
    Overall = Overall + c(i,i);
end
Overall = Overall/sum(c(:));
ConfusionMatrix = c %confusion matrix
