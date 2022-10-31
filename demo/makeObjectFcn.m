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
