function [newdata,visdata] = mySafeLevelSMOTE(data, minorityLabel, num2Add, options)
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

% safe-level of each minority class data
safeLevels = zeros(NofMinorityData,1);

% Save list of neighboring points for each minority data
nnarrays = cell(NofMinorityData,1);
for ii=1:NofMinorityData
    y = featuresMinority(ii,:); % a minority data
    [nnarray, ~] = knnsearch(featuresAll,y,'k',numNeighbors+1,...
        'Distance',distance, ...
        'SortIndices',true); % search for neighboring points
    % NOTE: this include self y, needs to omit y from nnarray
    nnarray = nnarray(2:end);
    
    idx = labelsAll(nnarray) == minorityLabel;
    NofNonMinority = sum(~idx); % number of non-minority data
    nnarrays{ii} = nnarray(idx); % keeps minority dataset only
    safeLevels(ii) = numNeighbors-NofNonMinority;
    % safe level of y, safeLevelP
    % safe level of nnarray, saveLevelN
end

% If the number of minority data is smaller than the requested number of new
% data set (num2Add), we randomly pick num2Add of minority data to be used to generate
% data.
if NofMinorityData >= num2Add
    idx = randperm(NofMinorityData,num2Add);
    featuresMinoritySubset = featuresMinority(idx,:);
    T1 = num2Add; % Number of data from minority dataset to be used
    T2 = 1; % Number of newdata from each minority dataset
else
    % Otherwise we use all minority data
    idx = randperm(NofMinorityData); % just to randamize
    featuresMinoritySubset = featuresMinority(idx,:);
    T1 = NofMinorityData; % Number of data from minority dataset to be used
    T2 = ceil(num2Add/NofMinorityData); % Number of newdata from each minority dataset
    % Note: doe to CEIL the total number of newdata may exceeds the
    % requested #, num2Add. Currently, the below has the routine to stop the process at num2Add.
end

% Array to save the synthesized features
newFeatures = zeros(num2Add,size(featuresMinority,2));
index = 1;

while index < num2Add 
% Make sure if num2Add is satisfied since it skips when gan = nan;
    for ii=1:T1 % Number of data from minority dataset to be used
        y = featuresMinoritySubset(ii,:); % a minority data
        safeLevelP = safeLevels(ii);
        
        [nnarray, ~] = knnsearch(featuresMinoritySubset,y,'k',numNeighbors+1,...
            'Distance',distance, ...
            'SortIndices',true); % search for neighboring points
        % NOTE: this include self y, needs to omit y from nnarray
        nnarray = nnarray(2:end);
        
        for kk=1:T2 % Number of newdata from each minority dataset
            nn = datasample(nnarray, 1); % pick one (randomly)
            safeLevelN = safeLevels(nn);
            diff = featuresMinoritySubset(nn,:) - y;
            gap = generateGap(safeLevelP, safeLevelN);
            if isnan(gap)
                continue; % for case 1
            end
            synthetic = y + gap.*diff; % “à‘}
            newFeatures(index,:) = synthetic;
            
            visdata{index,1} = y;
            visdata{index,2} = featuresMinoritySubset(nnarray,:);
            visdata{index,3} = featuresMinoritySubset(nn,:);
            visdata{index,4} = synthetic;
            
            index = index + 1;
            
            if index > num2Add
                break;
            end
        end
    end
end

% make newFeature to table data with the same variable names
tmp = array2table(newFeatures,'VariableNames',data.Properties.VariableNames(1:end-1));
% add label variable
newdata = addvars(tmp,repmat(minorityLabel,height(tmp),1),...
    'NewVariableNames',data.Properties.VariableNames(end));
end


function gap = generateGap(safeLevelP, safeLevelN)

if safeLevelN ~= 0
    safeLevelRatio = safeLevelP/safeLevelN;
else
    safeLevelRatio = inf;
end


if (isinf(safeLevelRatio) && safeLevelP == 0) % 1st case
    % When neighbors are all non-minority class (y and its neighbors)
    gap = nan;
    % does not generate positive synthetic instance
elseif (isinf(safeLevelRatio) && safeLevelP ~= 0) % 2nd case
    % When neighbors' neighbors are all non-minority, but there are
    % minority class around y
    gap = 0;
elseif (safeLevelRatio == 1) % 3rd case
    gap = rand();
elseif (safeLevelRatio > 1) % 4th case
    % When there are more minority class around y
    gap = rand()/safeLevelRatio;
elseif (safeLevelRatio < 1)
    % When there are more minority class around y's neighbor
    gap = rand()*safeLevelRatio + (1-safeLevelRatio);
else
    % warning('generateGap() in mySafeLevelSmote.m: something is wrong in getting gap');
    gap = rand();
end

end