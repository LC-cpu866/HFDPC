clear;
clc;

data = xlsread('D:\Xnewm\datasets\UCI\pima\pima.xlsx');
labels = xlsread('D:\Xnewm\datasets\UCI\pima\label.xlsx');

[n, m] = size(data);
numU = 10; % the number of clients
L_theta = 0.7;
theta = 0.7;
NCLUST = length(unique(labels));

tic;
% step 1：divide the dataset and create a ClientNew
users = cell(numU, 1);
for i = 1:numU
    item = zeros(n, m);
    label = zeros(n, 1);
    idx = i;
    count_idx = 1;
    item(count_idx,:) = data(idx,:);
    label(count_idx) = labels(idx);
    while idx+numU <= n
        count_idx = count_idx + 1;
        idx = idx + numU;
        item(count_idx,:) = data(idx,:);
        label(count_idx) = labels(idx);
    end
    X = item(1:count_idx,:);
    label = label(1:count_idx);
    users{i, 1} = ClientNew(X, label);
end

% step 2：distributed data dimensionality reduction
% "Research on distributed parallel dimensionality reduction algorithm based
% on PCA algorithm"
server = ServerNew();
if m>10
    users = server.DP_PCA(users);
end

% step 3：normalization of client data
users = server.Normalization(users);

% step 4：limit the density of the partitioned grid by the threshold L_theta
[server, L] = server.Meshing(users, L_theta, n, theta, NCLUST);

% step 5：assign points within the grid based on the assigned grid labels
for i = 1:numU
    users{i,1} = users{i,1}.Assign(server.grid, server.clG, L);
end
toc;

% step 6：plot results
data = users{1,1}.data;
labels = users{1,1}.labels;
cl = users{1,1}.cl;
for i = 2:numU
    data = [data; users{i,1}.data];
    labels = [labels; users{i,1}.labels];
    cl = [cl; users{i,1}.cl];
end
icl = ones(length(unique(cl)),1);
evaluation_index(labels, cl);