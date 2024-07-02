classdef SafeLevelSMOTE < SMOTE

    properties
        safeLevels
        nnarrays % 何に使う？？調査必要
    end

    methods(Access=protected)

        function calcAndSetParameters(obj)
            NofMinorityData = size(obj.featuresMinority,1);
            % safe-level of each minority class data
            obj.safeLevels = zeros(NofMinorityData,1);

            % Save list of neighboring points for each minority data
            obj.nnarrays = cell(NofMinorityData,1);
            for ii=1:NofMinorityData
                y = obj.featuresMinority(ii,:); % a minority data
                [nnarray, ~] = knnsearch(obj.featuresAll,y,'k',obj.NumNeighbors+1,...
                    'Distance',obj.distance, ...
                    'SortIndices',true); % search for neighboring points
                % NOTE: this include self y, needs to omit y from nnarray
                nnarray = nnarray(2:end);

                idx = obj.labelsAll(nnarray) == obj.minorityLabel;
                NofNonMinority = sum(~idx); % number of non-minority data
                obj.nnarrays{ii} = nnarray(idx); % keeps minority dataset only
                obj.safeLevels(ii) = obj.NumNeighbors-NofNonMinority;
                % safe level of y, safeLevelP
                % safe level of nnarray, saveLevelN
            end
        end

        function gain = calcGain(obj, ii, kk)
            safeLevelP = obj.safeLevels(ii);
            safeLevelN = obj.safeLevels(kk);
            gain = generateGap(safeLevelP, safeLevelN);
        end
    end
end


%% local function
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