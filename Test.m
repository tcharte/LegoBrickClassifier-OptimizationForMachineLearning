%% ECE 503 Course Project
% Todd Charter (V00853402)

%% Load Data
scale = 0.25;
image_width = 400*scale;
num_pixels = image_width^2;

listing = dir('dataset');
listing(1:2) = []; % Delete '.' and '..' file directories
N_samples = length(listing);
N_skip = 1;

X = [];
label = [];
for i = 1:N_skip:N_samples
    fname = listing(i).name;
    image = double(imread(strcat('./dataset/', fname)));
    image = imresize(image, scale);
    image = rgb2gray(image);
    X = [X reshape(image,num_pixels,1)];
    
    fname = fname(1:end-3); % remove '.jpg'
    parts = split(fname);
    label = [label; parts(1)];
    %image_info = part(end);
    
    fprintf('%d / %d\n', i, N_samples);
end

labels = unique(label);
y = arrayfun(@(x) find(string(labels) == string(x)), label); % true classes
D = [X; y'];

%% Extract HOG Features

H = [];
for i = 1:size(X,2)
    xi = X(:,i);
    mi = reshape(xi,image_width,image_width);
    hi = hog20(mi,7,9); % d = 7 and B = 9
    H = [H hi];
end
Dh = [H; y'];

save('data', D,Dh)

%% Split Data for Training/Testing

% Randomly Partition Data (train: 70%, test: 30%)
cv = cvpartition(size(X,2),'HoldOut',0.3);
idx = cv.test;

% Separate to training and test data
D_train = D(:,~idx);
D_test  = D(:,idx);

Dh_train = Dh(:,~idx);
Dh_test  = Dh(:,idx);

y_train = y(~idx);
y_test = y(idx);

%% Model Fitting
K = 50; % number of classes

[Ws,f]= SRMCC_bfgsML(D_train,'f_SRMCC','g_SRMCC',0.002,K,62);
[Whs,fh]= SRMCC_bfgsML(Dh_train,'f_SRMCC','g_SRMCC',0.001,K,57);


%% Prediction
Xhat_test = [D_train(1:end-1,:); ones(1,size(D_train,2))]; % replace last row with ones
Xhhat_test = [Dh_test(1:end-1,:); ones(1,size(Dh_test,2))]; % replace last row with ones

[~,ind_pre] = max((Xhat_test'*Ws)');
[~,ind_preh] = max((Xhhat_test'*Whs)');

%% Confusion Matrix

C = zeros(K,K);
for j = 1:K
 ind_j = find(ytest == j);
 for i = 1:K
 ind_pre_i = find(ind_pre == i);
 C(i,j) = length(intersect(ind_j,ind_pre_i));
 end
end 

Ch = zeros(K,K);
for j = 1:K
 ind_j = find(yhtest == j);
 for i = 1:K
 ind_pre_i = find(ind_preh == i);
 Ch(i,j) = length(intersect(ind_j,ind_pre_i));
 end
end 

accuracy = trace(C)/sum(C,'all')
accuracyh = trace(Ch)/sum(Ch,'all')