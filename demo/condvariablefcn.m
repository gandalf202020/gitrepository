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

