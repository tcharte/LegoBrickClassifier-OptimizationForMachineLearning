%% ECE 503 Course Project
% Todd Charter (V00853402)
run = '10class400samp100x100_eps1e-5';
%% Load Data
scale = 0.25;
image_width = 400*scale;
num_pixels = image_width^2;

listing = dir('dataset10class\');
listing(1:2) = []; % Delete '.' and '..' file directories
N_samples = length(listing);
N_skip = 4;

X = [];
label = [];
for i = 1:N_skip:N_samples/N_skip
    fname = listing(i).name;
    image = imread(strcat('./dataset10class/', fname));
    image = imresize(image, scale);
    image = rgb2gray(image);
    image = mat2gray(image,[0 255]);
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
tic;
H = [];
for i = 1:size(X,2)
    xi = X(:,i);
    mi = reshape(xi,image_width,image_width);
    hi = hog20(mi,7,9); % d = 7 and B = 9
    H = [H hi];
end
Dh = [H; y'];
hog_extraction_time = toc;

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
K = 10; % number of classes
epsi = 1e-5;

% Train pixel based models
tic;
[Ws_bfgs,f_bfgs,k_bfgs] = cg(D_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
t_train_bfgs = toc;
tic;
[Ws_bfgsML,f_bfgsML,k_bfgsML]= SRMCC_bfgsML(D_train,'f_SRMCC','g_SRMCC',0.002,K,epsi);
t_train_bfgsML = toc;
tic;
[Ws_cg,f_cg,k_cg] = cg(D_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
t_train_cg = toc;
tic;
[Ws_grad_desc,f_grad_desc,k_grad_desc] = grad_desc(D_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
t_train_grad_desc = toc;

% Train HOG models
tic;
[Whs_bfgs,fh_bfgs,kh_bfgs] = cg(Dh_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
th_train_bfgs = toc;
tic;
[Whs_bfgsML,fh_bfgsML,kh_bfgsML]= SRMCC_bfgsML(Dh_train,'f_SRMCC','g_SRMCC',0.001,K,epsi);
th_train_bfgsML = toc;
tic;
[Whs_cg,fh_cg,kh_cg] = cg(Dh_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
th_train_cg = toc;
tic;
[Whs_grad_desc,fh_grad_desc,kh_grad_desc] = grad_desc(Dh_train,'f_SRMCC','g_SRMCC',mu,K,epsi);
th_train_grad_desc = toc;



%% Prediction
Xhat_test = [D_test(1:end-1,:); ones(1,size(D_test,2))]; % replace last row with ones
Xhhat_test = [Dh_test(1:end-1,:); ones(1,size(Dh_test,2))]; % replace last row with ones

% Run pixel based models
tic;
[~,ind_pre_bfgs] = max((Xhat_test'*Ws_bfgs));
t_pred_bfgs = toc;
tic;
[~,ind_pre_bfgsML] = max((Xhat_test'*Ws_bfgsML));
t_pred_bfgsML = toc;
tic;
[~,ind_pre_cg] = max((Xhat_test'*Ws_cg));
t_pred_cg = toc;
tic;
[~,ind_pre_grad_desc] = max((Xhat_test'*Ws_grad_desc));
t_pred_grad_desc = toc;

% Run HOG based models
tic;
[~,ind_preh_bfgs] = max((Xhhat_test'*Whs_bfgs));
th_pred_bfgs = toc;
tic;
[~,ind_preh_bfgsML] = max((Xhhat_test'*Whs_bfgsML));
th_pred_bfgsML = toc;
tic;
[~,ind_preh_cg] = max((Xhhat_test'*Whs_cg));
th_pred_cg = toc;
tic;
[~,ind_preh_grad_desc] = max((Xhhat_test'*Whs_grad_desc));
th_pred_grad_desc = toc;

%% Evaluation

% Pixel Based Confusion Matrices
figure
Cm = confusionchart(y_test, ind_pre_bfgs);
Cm.title('Confusion Matrix for Pixel Based Classifier with BFGS')
savefig(strcat('./images/Cm_', run, '_bfgs.fig'))
saveas(gcf, strcat('./images/Cm_', run, '_bfgs.fig'))
close

figure
Cm = confusionchart(y_test, ind_pre_bfgsML);
Cm.title('Confusion Matrix for Pixel Based Classifier with MLBFGS')
savefig(strcat('./images/Cm_', run, '_mlbfgs.fig'))
saveas(gcf, strcat('./images/Cm_', run, '_mlbfgs.fig'))
close

figure
Cm = confusionchart(y_test, ind_pre_cg);
Cm.title('Confusion Matrix for Pixel Based Classifier with CG')
savefig(strcat('./images/Cm_', run, '_cg.fig'))
saveas(gcf, strcat('./images/Cm_', run, '_cg.fig'))
close

figure
Cm = confusionchart(y_test, ind_pre_grad_desc);
Cm.title('Confusion Matrix for Pixel Based Classifier with GRAD DESC')
savefig(strcat('./images/Cm_', run, '_grad_desc.fig'))
saveas(gcf, strcat('./images/Cm_', run, '_grad_desc.fig'))
close


% HOG Confusion Matrices
figure
Cmh = confusionchart(y_test, ind_preh_bfgs);
Cmh.title('Confusion Matrix for Classifier with HOG and BFGS')
savefig(strcat('./images/Cmh_', run, '_bfgs.fig'))
saveas(gcf, strcat('./images/Cmh_', run, '_bfgs.fig'))
close

figure
Cmh = confusionchart(y_test, ind_preh_bfgsML);
Cmh.title('Confusion Matrix for Classifier with HOG and MLBFGS')
savefig(strcat('./images/Cmh_', run, '_mlbfgs.fig'))
saveas(gcf, strcat('./images/Cmh_', run, '_mlbfgs.fig'))
close

figure
Cmh = confusionchart(y_test, ind_preh_cg);
Cmh.title('Confusion Matrix for Classifier with HOG and CG')
savefig(strcat('./images/Cmh_', run, '_cg.fig'))
saveas(gcf, strcat('./images/Cmh_', run, '_cg.fig'))
close

figure
Cmh = confusionchart(y_test, ind_preh_grad_desc);
Cmh.title('Confusion Matrix for Classifier with HOG and GRAD DESC')
savefig(strcat('./images/Cmh_', run, '_grad_desc.fig'))
saveas(gcf, strcat('./images/Cmh_', run, '_grad_desc.fig'))
close

% Accuracies for pixel based models
acc_bfgs = sum(y_test==ind_pre_bfgs)/numel(y_test)
acc_mlbfgs = sum(y_test==ind_pre_bfgsML)/numel(y_test)
acc_cg = sum(y_test==ind_pre_cg)/numel(y_test)
acc_grad_desc = sum(y_test==ind_pre_grad_desc)/numel(y_test)

% Accuracies for HOG models
acch_bfgs = sum(y_test==ind_preh_bfgs)/numel(y_test)
acch_mlbfgs = sum(y_test==ind_preh_bfgsML)/numel(y_test)
acch_cg = sum(y_test==ind_preh_cg)/numel(y_test)
acch_grad_desc = sum(y_test==ind_preh_grad_desc)/numel(y_test)

save(strcat('data_', run), '-v7.3')