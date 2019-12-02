% Custom validation function
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