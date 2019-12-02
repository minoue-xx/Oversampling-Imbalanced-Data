function [newdata,visdata] = mySafeLevelSMOTE(data, minorityLabel, N, k)
% Input
% data: table data with features and labels
% 1. The right-most variable is treated as labels
% TODO: if not so, move var to the right-most
% 2. Features are expected to be numeric values
% TODO: Validation routine
%
% minorityLabel (scalar string): Label to oversample
% N (scalar numeric): Number of data to generate
% k (scalar integer): number of neighbors to consider
%
% Output
% newdata: generated dataset
% visdata: optional output for debugging
%-------------------------------------------------------------------------
% Copyright (c) 2019 Michio Inoue

arguments
    data {mustBeTableWithClassname}
    minorityLabel (1,1) string 
    N (1,1) double {mustBeNonnegative, mustBeInteger}
    k (1,1) double {mustBePositive, mustBeInteger} = 5
end

% If N is smaller than zero, do not oversample data
if  N <= 0
    newdata = table;
    visdata = cell(1);
    return;
end

visdata = cell(N,4);
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
    [nnarray, ~] = knnsearch(featuresAll,y,'k',k+1,'SortIndices',true);  % search for neighboring points
    % NOTE: this include self y, needs to omit y from nnarray
    nnarray = nnarray(2:end);
    
    idx = labelsAll(nnarray) == minorityLabel;
    NofNonMinority = sum(~idx); % number of non-minority data
    nnarrays{ii} = nnarray(idx); % keeps minority dataset only
    safeLevels(ii) = k-NofNonMinority;
    % safe level of y, safeLevelP
    % safe level of nnarray, saveLevelN
end

% If the number of minority data is smaller than the requested number of new
% data set (N), we randomly pick N of minority data to be used to generate
% data.
if NofMinorityData >= N
    idx = randperm(NofMinorityData,N);
    featuresMinoritySubset = featuresMinority(idx,:);
    T1 = N; % Number of data from minority dataset to be used
    T2 = 1; % Number of newdata from each minority dataset
else
    % Otherwise we use all minority data
    idx = randperm(NofMinorityData); % just to randamize
    featuresMinoritySubset = featuresMinority(idx,:);
    T1 = NofMinorityData; % Number of data from minority dataset to be used
    T2 = ceil(N/NofMinorityData); % Number of newdata from each minority dataset
    % Note: doe to CEIL the total number of newdata may exceeds the
    % requested #, N. Currently, the below has the routine to stop the process at N.
end

% Array to save the synthesized features
newFeatures = zeros(N,size(featuresMinority,2));
index = 1;

for ii=1:T1 % Number of data from minority dataset to be used
    y = featuresMinoritySubset(ii,:); % a minority data
    safeLevelP = safeLevels(ii);
    
    [nnarray, ~] = knnsearch(featuresMinoritySubset,y,'k',k+1,'SortIndices',true); % search for neighboring points
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
        
        if index > N
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


function gap = generateGap(safeLevelP, safeLevelN)

if safeLevelN ~= 0
    safeLevelRatio = safeLevelP/safeLevelN;
else
    safeLevelRatio = inf;
end

if (isinf(safeLevelRatio) && safeLevelP == 0) % 1st case
    gap = nan;
    % does not generate positive synthetic instance
elseif (isinf(safeLevelRatio) && safeLevelP ~= 0) % 2nd case
    gap = 0;
elseif (safeLevelRatio == 1) % 3rd case
    gap = rand();
elseif (safeLevelRatio > 1) % 4th case
    gap = rand()/safeLevelRatio;
elseif (safeLevelRatio < 1)
    gap = rand()*safeLevelRatio + (1-safeLevelRatio);
else
    disp('something is wrong in getting gap');
    gap = rand();
end

end