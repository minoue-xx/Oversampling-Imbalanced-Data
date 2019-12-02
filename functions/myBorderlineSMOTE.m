function [newdata,visdata] = myBorderlineSMOTE(data, minorityLabel, num2Add, options)
% Input
% data: table data with features and labels
% 1. The right-most variable is treated as labels
% 2. Features are expected to be numeric values
%
% minorityLabel (scalar string): Label to oversample
% num2Add (scalar numeric): Number of data to generate
% options.NumNeighbors (scalar integer): number of neighbors to consider
% options.Standardize (scalar logical):
% Standard-euclidean (true) or Euclidean distance (false) distance to search the neighbors
%
% Output
% newdata: generated dataset
% visdata: optional output for debugging
%-------------------------------------------------------------------------
% Copyright (c) 2019 Michio Inoue

arguments
    data {mustBeTableWithClassname}
    minorityLabel (1,1) string 
    num2Add (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    options.NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 5
    options.Standardize (1,1) logical = false;
end

numNeighbors = options.NumNeighbors;
if options.Standardize
    distance = 'seuclidean';
else
    distance = 'euclidean';
end

% If N is smaller than zero, do not oversample data
if  num2Add <= 0
    newdata = table;
    visdata = cell(1);
    return;
end

visdata = cell(num2Add,4);
% Optional output for visualization purpose only
% 1: y, 2: nnarray, 3: y2, 4: synthetic

% labels of whote dataset
labelsAll = string(data{:,end});

% all feature dataset
featuresAll = data{:,1:end-1};
% feature dataset of the minority label
featuresMinority = data{labelsAll == minorityLabel,1:end-1};

% Number of minority data
NofMinorityData = size(featuresMinority,1);

% Number of synthetic data to generate is proportional to weights
weights = zeros(NofMinorityData,1);
% Save list of neighboring points for each minority data
nnarrays = cell(NofMinorityData,1);

for ii=1:NofMinorityData
    y = featuresMinority(ii,:); % a minority data
    [nnarray, ~] = knnsearch(featuresAll,y,'k',numNeighbors+1,...
        'Distance',distance, ...
        'SortIndices',true); % search for neighboring points
    % NOTE: this include self y, needs to omit y from nnarray
    nnarray = nnarray(2:end);
    % Note: nnarray will have a list of index of each neighboring points
    % witin the all dataset (not within the minority subset)
    idx = labelsAll(nnarray) == minorityLabel;
    NofNonMinority = sum(~idx); % number of non-minority data
    nnarrays{ii} = nnarray(idx); % keeps minority dataset only
    weights(ii) = NofNonMinority; % keeps the ratio of non-minority dataset
end

% So who's on borderline?
isdanger = weights < numNeighbors & weights >= numNeighbors/2;

if all(~isdanger)
    % callsmote instead
    %     disp('calling SMOTE instead');
    %     newFeatures = mySMOTE(data, minorityLabel, N, k);
else
    % Decide the number of synthetic data to genarate for each minority
    % dataset
    NofDanger = sum(isdanger);
    N2generate = ceil(num2Add/NofDanger);
    
    newFeatures = zeros(num2Add,size(featuresAll,2));
    index = 1;
    
    featuresDangered = featuresMinority(isdanger,:);
    for ii=1:NofDanger % for all the endangered data
        y = featuresDangered(ii,:); % a minority and dangered data
        [nnarray, ~] = knnsearch(featuresMinority,y,'k',numNeighbors+1,...
            'Distance',distance, ...
            'SortIndices',true); % search for neighboring points
        % NOTE: this include self y, needs to omit y from nnarray
        nnarray = nnarray(2:end);
        
        for kk=1:N2generate % generate N2generate of synthetic data
            
            nn = datasample(nnarray, 1); % pick one (randomly)
            % Interpolation
            diff = featuresMinority(nn,:) - y;
            synthetic = y + rand.*diff;
            newFeatures(index,:) = synthetic;
            
            visdata{index,1} = y;
            visdata{index,2} = featuresMinority(nnarray,:);
            visdata{index,3} = featuresMinority(nn,:);
            visdata{index,4} = synthetic;
            
            index = index + 1;
            
            if index > num2Add
                break;
            end
        end
    end
    
    % make newFeature to table data with the same variable names
    tmp = array2table(newFeatures,'VariableNames',data.Properties.VariableNames(1:end-1));
    % add label variable
    newdata = addvars(tmp,repmat(minorityLabel,height(tmp),1),...
        'NewVariableNames',data.Properties.VariableNames(end));
end