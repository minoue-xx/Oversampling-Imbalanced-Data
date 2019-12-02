% Custom validation function
% https://jp.mathworks.com/help/matlab/matlab_prog/function-argument-validation-1.html
%-------------------------------------------------------------------------
% Copyright (c) 2019 Michio Inoue
function mustBeTableWithClassname(arg)
    features = arg{:,1:end-1};
    class = arg{:,end};
    if ~isnumeric(features)
        error(['not numeri features'])
    end
    if ~isstring(class)
        error(['not a string classname'])
    end
end