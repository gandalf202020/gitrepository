clear;
clc;
% close all;
start_time=tic; %Timer
rng(42)
global num_layers_set
%%
data_folder = 'Data'; 
input_filename_lfm = '25bar8D_LFM'; % input file
%% Read input Data
% Read data
in_num = 8;
out_num = 9;
j = 900;

DATA_lfm = readtable(fullfile(data_folder,input_filename_lfm));
DATA_lfm_input = table2array(DATA_lfm(:,1:in_num));
DATA_lfm_target = table2array(DATA_lfm(:,out_num));
DATA = [DATA_lfm_input DATA_lfm_target];
DATA = DATA(all(~isnan(DATA),2),:);

data = DATA(1:j,1:in_num);
target = DATA(1:j,out_num);
%% BO1
%% User inputs
val_perc = 0.15; %percentage of data for validation set
test_perc = 0.15; %percentage of data for test test

% Set Hyperparameters
maxEpochs = 500; %stopping criteria - max training epochs
GradientThreshold = 1; %gradient clipping threshold
ValidationPatience = 5; %stopping criteria - number of epochs with 
                                       %increasing validation loss rate
LearnRateDropFactor = 0.5; %learning rate drop multiplier
leaky_epsilon = 0.1; %leak rate for negative size of leaky Relu layers
% Hyperparameter ranges
FC_1_min=5;FC_1_max=50;
FC_2_min=5;FC_2_max=50;
FC_3_min=5;FC_3_max=50;
FC_4_min=5;FC_4_max=50;
FC_5_min=5;FC_5_max=50;
FC_6_min=5;FC_6_max=50;
FC_7_min=5;FC_7_max=50;
FC_8_min=5;FC_8_max=50;
FC_9_min=5;FC_9_max=50;
FC_10_min=5;FC_10_max=50;

num_layers_set = [1,2,3,4,5,6,7,8,9,10];

InitialLearnRate_min = 0.01; %min initial learn rate for optimizer
InitialLearnRate_max = 1; %max initial learn rate for optimizer
batch_size_set = [4,8,16,32,64,128,256,512]; % batch size during training
ind=randi(numel(batch_size_set),1,1);
L2Regularization_min = 1e-10;
L2Regularization_max = 1e-2;
Momentum_min = 0.8;
Momentum_max = 0.99;

%% Prepare data for modeling
% Transpose because MATLAB likes features as rows for neural nets
X = data';
t = target';
[x_scaled,PX] = mapminmax(X);
% Divide data into train, validation, and test sets
[trainInd,valInd,testInd] = dividerand(size(X,2), ...
    (1 - val_perc - test_perc), val_perc, test_perc);
x_train = x_scaled(:,trainInd);
t_train = t(:,trainInd);
x_val = x_scaled(:,valInd);
t_val = t(:,valInd);
x_test = x_scaled(:,testInd);
t_test = t(:,testInd);
% Get the number of samples in the training data.
nFeatures = size(x_train,1);
% Number of output features
nResponses = size(t_train,1);
% Number of samples in the train, validation, and test sets
nSamples = size(x_train,2);
nValSamples = size(x_val,2);
nTestSamples = size(x_test,2);
Xtrain = reshape(x_train, [1,1,nFeatures,nSamples]);
Xval = reshape(x_val, [1,1,nFeatures,nValSamples]);
Xtest = reshape(x_test, [1,1,nFeatures,nTestSamples]);
%% Prepare Bayesian Optimization
% Variablevariable = optimizableVariable (Name,Range,Name,Value) 
optimVars = [
    optimizableVariable('FC_1',[FC_1_min FC_1_max],'Type','integer')
    optimizableVariable('FC_2',[FC_2_min FC_2_max],'Type','integer')
    optimizableVariable('FC_3',[FC_3_min FC_3_max],'Type','integer')
    optimizableVariable('FC_4',[FC_4_min FC_4_max],'Type','integer')
    optimizableVariable('FC_5',[FC_5_min FC_5_max],'Type','integer')
    optimizableVariable('FC_6',[FC_6_min FC_6_max],'Type','integer')
    optimizableVariable('FC_7',[FC_7_min FC_7_max],'Type','integer') 
    optimizableVariable('FC_8',[FC_8_min FC_8_max],'Type','integer')
    optimizableVariable('FC_9',[FC_9_min FC_9_max],'Type','integer')
    optimizableVariable('FC_10',[FC_10_min FC_10_max],'Type','integer')   
    optimizableVariable('num_layers_set',[1 length(num_layers_set)],...
        'Type','integer')
    optimizableVariable('InitialLearnRate',...
        [InitialLearnRate_min InitialLearnRate_max],'Transform','log')
    optimizableVariable('Momentum',[Momentum_min Momentum_max])
    optimizableVariable('solverName',{'sgdm' 'rmsprop' 'adam'},...
        'Type','categorical')
    optimizableVariable('batch_size',{'4' '16' '32' '64' '128' '256' '512'},...
        'Type','categorical')
    optimizableVariable('L2Regularization',...
        [L2Regularization_min L2Regularization_max],'Transform','log')];
    
ObjFcn = makeObjFcn(Xtrain, t_train, Xval, t_val, nFeatures, nResponses,...
    batch_size_set, num_layers_set, maxEpochs, GradientThreshold,...
    ValidationPatience, LearnRateDropFactor, leaky_epsilon);

% MaxObjectiveEvaluations = 30;% ‘MaxObjectiveEvaluations’—目标函数评估限制
% UseParallel = false;
% PlotFcn = {@plotObjectiveModel,@plotMinObjective}; %'all',[]
% AcquisitionFunctionName = 'expected-improvement-plus';
opt_num = 10;%50
BayesObject = bayesopt(ObjFcn,optimVars, ...
    'ConditionalVariableFcn',@condvariablefcn, ...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'MaxObjectiveEvaluations',opt_num, ...
    'IsObjectiveDeterministic',false, ... 
    'ExplorationRatio',0.2,...
    'GPActiveSetSize',80,... 
    'UseParallel',false,...
    'ParallelMethod','max-observed',...
    'NumSeedPoints',10,...
    'MaxTime',3600,...
    'Verbose',2,...
    'PlotFcn','all');
%%
% subplot(1,1,1)
x = [1:1:opt_num];
y = BayesObject.ObjectiveMinimumTrace;
semilogy(x,y,'linewidth',2);
title('BO');
legend('最小目标函数');
xlim([0 opt_num]);

%% Use best parameters to train model and get results
bestIdx = BayesObject.IndexOfMinimumTrace(end);
best_inputs = BayesObject.XTrace(bestIdx,:);

% Save best inputs
writetable(best_inputs, 'Models/Best_NN_Parameters.csv')

% Best Hyperparameters!
FC_1 = best_inputs.FC_1; %number of nodes in first hidden layer
FC_2 = best_inputs.FC_2; %number of nodes in second hidden layer
FC_3 = best_inputs.FC_3; %number of nodes in third hidden layer
FC_4 = best_inputs.FC_4; %number of nodes in third hidden layer
FC_5 = best_inputs.FC_5; %number of nodes in third hidden layer
FC_6 = best_inputs.FC_6; %number of nodes in third hidden layer
FC_7 = best_inputs.FC_7; %number of nodes in third hidden layer
FC_8 = best_inputs.FC_8; %number of nodes in third hidden layer
FC_9 = best_inputs.FC_9; %number of nodes in third hidden layer
FC_10 = best_inputs.FC_10; %number of nodes in third hidden layer
num_layers_best = best_inputs.num_layers_set; %Index of best num layers
InitialLearnRate = best_inputs.InitialLearnRate; %initial learn rate
batch_size_best = best_inputs.batch_size; % Index of best batch size
L2Regularization = best_inputs.L2Regularization; %Regularization weighting
best_solverName = best_inputs.solverName;
Momentum = best_inputs.Momentum;

%% Create the DNN_L
data_folder = 'Data'; 
input_filename_lfm = '25bar8D_LFM'; % input file
input_filename_hfm = '25bar8D_HFM'; % input file
%% Read input Data
% Read data
in_dim = 8;
out_dim = 9;
j = 500;     %lfm number
DATA_lfm = readtable(fullfile(data_folder,input_filename_lfm));
DATA_lfm_input = table2array(DATA_lfm(:,1:in_dim));
DATA_lfm_target = table2array(DATA_lfm(:,out_dim));
DATAlfm = [DATA_lfm_input DATA_lfm_target];

p = DATAlfm(1:j,1:in_dim)';
t = DATAlfm(1:j,out_dim)';
%normalization
[pn,ps]=mapminmax(p,0,1);
[tn,ts]=mapminmax(t,0,1);
%% 
Node1=FC_1;   % 隐层第一层节点数 
Node2=FC_2;   % 隐层第二层节点数   
Node3=FC_3;
Node4=FC_4;
Node5=FC_5;
% Node6=FC_6;
% Node7=FC_7;
% Node8=FC_8;
% Node9=FC_9;
% Node10=FC_10;
TypeNum = 1;   % 输出维数   
jihuo1 = 'tansig';  %激活函数
jihuo2 = 'tansig';  
jihuo3 = 'tansig'; 
jihuo4 = 'tansig';
jihuo5 = 'tansig';
jihuo6 = 'tansig';
% jihuo7 = 'poslin';
% jihuo8 = 'poslin';
% jihuo9 = 'poslin';
% jihuo10 = 'poslin';
% jihuo11 = 'poslin';
% 'sigmoid'
% 'tanh'
% 'poslin'--relu
% 'tansig'
Node = [Node1,Node2,Node3,Node4,Node5];
jihuo = {jihuo1 jihuo2 jihuo3 jihuo4 jihuo5 jihuo6};
net=newff(minmax(pn),[Node,...
    TypeNum],jihuo,'trainlm');
% traingd
% traingda
% traingdx
% trainlm

net.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%Settings, number of hidden layers, number of nodes, activation function, learning algorithm
net.trainParam.show=200;
net.trainParam.goal=1e-7;    %The accuracy to be achieved by training
net.trainParam.lr=InitialLearnRate;      %Learning rate 
net.trainParam.epochs=101;%Training times setting
net.trainParam.max_fail=30;%Maximum number of non-falling steps
net.trainParam.lr=10^(-2);%learning rate 
net.trainParam.mc=Momentum;%Momentum factor
net.trainParam.min_grad=1.00e-10;%gradient

[net,info]=train(net,pn,tn);        %Training net

%% HFM pretrained result
% data_folder = 'Data'; 
% input_filename_hfm = '20220421_2d_10bar_hfm'; % input file
% Read input Data
% Read data
DATA_hfm = readtable(fullfile(data_folder,input_filename_hfm));
DATA_hfm_input = table2array(DATA_hfm(:,1:in_dim));
DATA_hfm_target = table2array(DATA_hfm(:,out_dim));
% DATA = [DATA_lfm_input DATA_lfm_target DATA_hfm_target];
DATAhfm = [DATA_lfm_input DATA_lfm_target];
DATAhfm = DATAhfm(all(~isnan(DATAhfm),2),:);

phfm = DATAhfm(1:j,1:in_dim)';
thfm = DATAhfm(1:j,out_dim)';
[pnhfm,pshfm]=mapminmax(phfm,0,1);
[tnhfm,tshfm]=mapminmax(thfm,0,1);
for i=1:500
layer1(:,i)=tansig(net.IW{1,1}*pnhfm(:,i)+net.b{1});
layer2(:,i)=tansig(net.LW{2,1}*layer1(:,i)+net.b{2});
layer3(:,i)=tansig(net.LW{3,2}*layer2(:,i)+net.b{3});
layer4(:,i)=tansig(net.LW{4,3}*layer3(:,i)+net.b{4});
layer5(:,i)=tansig(net.LW{5,4}*layer4(:,i)+net.b{5});
end
%% Finish program
%display total time to complete tasks
tElapsed = toc(start_time); 
hour=floor(tElapsed/3600);
tRemain = tElapsed - hour*3600;
min=floor(tRemain/60);
sec = tRemain - min*60;
 
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Program Complete!!!!!')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')
% Display regression accuracy metrics
disp(['Time to complete: ',num2str(hour),' hours, ',...
    num2str(min),' minutes, ',num2str(sec),' seconds'])

function ObjFcn = makeObjFcn(Xtrain, t_train, Xval, t_val, nFeatures,...
    nResponses, batch_size_set, num_layers_set, maxEpochs, ...
    GradientThreshold, ValidationPatience, LearnRateDropFactor,...
    leaky_epsilon)
    
    ObjFcn = @valErrorFun;

    function [RMSE_val] = valErrorFun(optVars)
        
        %% Build the Deep Neural Net (NN) model
        % Create the deep NN. This starts with an imageInputLayer as the 
        % data input layer. This is followed by a number of hidden layers.
        % Each hidden layer uses a fullyConnectedLayer and an activation 
        % function. The fullyConnectedLayer multiplies the outputs of the
        % pervious layer by a weight, adds those weighted inputs, and 
        % applies a bias. It does not have an inherent activation function
        % (besides the linear activation). The activation functions used 
        % below are all leaky rectified linear units (Leaky Relu). The 
        % final layer, regressionLayer, calculates the loss function for
        % use in the back propagation process. In this case, the loss
        % function is mean squared error (MSE).
        
        num_layers = num_layers_set(optVars.num_layers_set);
        
        if num_layers == 10
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(optVars.FC_6,'Name','FC6')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu6')
                fullyConnectedLayer(optVars.FC_7,'Name','FC7')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu7')
                fullyConnectedLayer(optVars.FC_8,'Name','FC8')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu8')
                fullyConnectedLayer(optVars.FC_9,'Name','FC9')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu9')
                fullyConnectedLayer(optVars.FC_10,'Name','FC10')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu10')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 9
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(optVars.FC_6,'Name','FC6')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu6')
                fullyConnectedLayer(optVars.FC_7,'Name','FC7')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu7')
                fullyConnectedLayer(optVars.FC_8,'Name','FC8')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu8')
                fullyConnectedLayer(optVars.FC_9,'Name','FC9')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu9')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 8
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(optVars.FC_6,'Name','FC6')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu6')
                fullyConnectedLayer(optVars.FC_7,'Name','FC7')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu7')
                fullyConnectedLayer(optVars.FC_8,'Name','FC8')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu8')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 7
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(optVars.FC_6,'Name','FC6')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu6')
                fullyConnectedLayer(optVars.FC_7,'Name','FC7')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu7')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 6
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(optVars.FC_6,'Name','FC6')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu6')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 5
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(optVars.FC_5,'Name','FC5')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu5')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 4
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(optVars.FC_4,'Name','FC4')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu4')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];            
        elseif num_layers == 3
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
                fullyConnectedLayer(optVars.FC_3,'Name','FC3')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        elseif num_layers == 2
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
                fullyConnectedLayer(optVars.FC_2,'Name','FC2')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        else
            layers = [ ...
                imageInputLayer([1 1 nFeatures],'Name','Input')
                fullyConnectedLayer(optVars.FC_1,'Name','FC1')
                leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
                fullyConnectedLayer(nResponses, 'Name', 'FC_output')
                regressionLayer('Name', 'Output')];
        end
        
        
        %% Train the Deep Neural Net (NN) model
        % Validation frequency is how often to check the loss of the 
        % validation set (expressed in iterations (samples)).
        batch_size = batch_size_set(optVars.batch_size);
        
        validationFrequency = floor(size(t_train,2)/batch_size);
        LearnRateDropPeriod = floor(maxEpochs/10);
        
        solverName = cellstr(optVars.solverName);
        
        options = trainingOptions(solverName{1}, ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',batch_size, ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',LearnRateDropPeriod, ...
            'LearnRateDropFactor',LearnRateDropFactor, ...
            'GradientThreshold',GradientThreshold, ...
            'L2Regularization',optVars.L2Regularization, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{Xval,t_val'}, ...
            'ValidationFrequency',validationFrequency, ...
            'ValidationPatience',ValidationPatience, ...
            'Verbose',0,...
            'VerboseFrequency',validationFrequency,...
            'Plots','none');
        
        solverName = cellstr(optVars.solverName);
        if strcmp(solverName{1}, 'sgdm') == 1
            options.Momentum = optVars.Momentum;
        end
        
        % Train the network 
        [net,info] = trainNetwork(Xtrain,t_train',layers,options);
        
        % Predict validation and check results
        y_hat_val = predict(net,Xval);
        
        res_val = y_hat_val - t_val';
        RMSE_val = sqrt(mean(res_val.^2));
        
    end
end


function Xnew = condvariablefcn(X)
    global num_layers_set
    
    Xnew = X;
    
    num_layers = num_layers_set(X.num_layers_set);
    
      if num_layers ==9
        Xnew.FC_10 = 0;
      
     elseif num_layers ==8
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        
     elseif num_layers ==7
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;

     elseif num_layers ==6
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;

     elseif num_layers ==5
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;
        Xnew.FC_6 = 0;  

     elseif num_layers ==4
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;
        Xnew.FC_6 = 0;
        Xnew.FC_5 = 0;

    elseif num_layers ==3
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;
        Xnew.FC_6 = 0;
        Xnew.FC_5 = 0;
        Xnew.FC_4 = 0;    
    elseif num_layers == 2
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;
        Xnew.FC_6 = 0;
        Xnew.FC_5 = 0;
        Xnew.FC_4 = 0;  
        Xnew.FC_3 = 0;
    elseif num_layers ==1
        Xnew.FC_10 = 0;
        Xnew.FC_9 = 0;
        Xnew.FC_8 = 0;
        Xnew.FC_7 = 0;
        Xnew.FC_6 = 0;
        Xnew.FC_5 = 0;
        Xnew.FC_4 = 0;  
        Xnew.FC_3 = 0;
        Xnew.FC_2 = 0;
    end
    
    %solverName = cellstr(X.solverName);
    %if strcmp(solverName{1}, 'sgdm') ~= 1
    %    Xnew.Momentum = 0;
    %end
end



