start_time=tic; %Timer
rng(42)
global num_layers_set
%%
data_folder = 'Data'; 
input_filename_hfm = '25bar8D_HFM'; % input file
%% Read input Data
% Read data
in_num = 8;
out_num = 9;
j = 500;
layernum = 'layer5';
DATA_hfm = readtable(fullfile(data_folder,input_filename_lfm));
DATA_hfm_input = layernum;
DATA_hfm_target = table2array(DATA_lfm(:,out_num));
DATA = [DATA_lfm_input DATA_lfm_target];
DATA = DATA(all(~isnan(DATA),2),:);

data = DATA(1:j,1:in_num);
target = DATA(1:j,out_num);
%% BO2
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
FC_1_min = 1; %min number of nodes in first hidden layer
FC_1_max = 20; %min number of nodes in first hidden layer
FC_2_min = 1; %min number of nodes in second hidden layer
FC_2_max = 20; %min number of nodes in first hidden layer
FC_3_min = 1; %min number of nodes in second hidden layer
FC_3_max = 20; %min number of nodes in first hidden layer
num_layers_set = [1,2,3];
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
opt_num = 50;%50
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
num_layers_best = best_inputs.num_layers_set; %Index of best num layers
InitialLearnRate = best_inputs.InitialLearnRate; %initial learn rate
batch_size_best = best_inputs.batch_size; % Index of best batch size
L2Regularization = best_inputs.L2Regularization; %Regularization weighting
best_solverName = best_inputs.solverName;
Momentum = best_inputs.Momentum;

%% Create the DNN_H
% Read input Data
p = DATA(1:j,1:in_dim)';
t = DATA(1:j,out_dim)';
%normalization
[pn,ps]=mapminmax(p,0,1);
[tn,ts]=mapminmax(t,0,1);
%% 
Node1=FC_1;   % 隐层第一层节点数 
Node2=FC_2;   % 隐层第二层节点数   

TypeNum = 1;   % 输出维数   
jihuo1 = 'tansig';  %激活函数
jihuo2 = 'tansig';  
jihuo3 = 'tansig'; 

% 'sigmoid'
% 'tanh'
% 'poslin'--relu
% 'tansig'
Node = [Node1,Node2];
jihuo = {jihuo1 jihuo2 jihuo3};
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
%设置、隐层数量、节点数、激活函数、学习算法
net.trainParam.show=200;
net.trainParam.goal=1e-7;    %训练所要达到的精度 
net.trainParam.lr=InitialLearnRate;      %学习速率 
net.trainParam.epochs=101;%训练次数设置
net.trainParam.max_fail=30;%最大不下降步数
net.trainParam.lr=10^(-2);%学习率设置,应设置为较少值，太大虽然会在开始加快收敛速度，但临近最佳点时，会产生动荡，而致使无法收敛
net.trainParam.mc=Momentum;%动量因子的设置，默认为0.9
net.trainParam.min_grad=1.00e-10;%gradient
[net,info]=train(net,pn,tn);        %训练net

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
        
        if num_layers == 3
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
    
    if num_layers == 2
        Xnew.FC_3 = 0;
    elseif num_layers ==1
        Xnew.FC_3 = 0;
        Xnew.FC_2 = 0;
    end
    
    %solverName = cellstr(X.solverName);
    %if strcmp(solverName{1}, 'sgdm') ~= 1
    %    Xnew.Momentum = 0;
    %end

end



