classdef ADASYN < SMOTE

    properties
        weights
        nnarrays % 何に使う？？調査必要
    end

    methods(Access=protected)

        function calcAndSetParameters(obj)
            NofMinorityData = size(obj.featuresMinority,1);
            % Number of synthetic data to generate is proportional to weights
            obj.weights = zeros(NofMinorityData,1);
            % Save list of neighboring points for each minority data
            obj.nnarrays = cell(NofMinorityData,1);

            for ii=1:NofMinorityData
                y = obj.featuresMinority(ii,:); % a minority data
                [nnarray, ~] = knnsearch(obj.featuresAll,y,'k',obj.NumNeighbors+1,...
                    'Distance',obj.distance, ...
                    'SortIndices',true); % search for neighboring points
                % NOTE: this include self y, needs to omit y from nnarray
                nnarray = nnarray(2:end);
                % Note: nnarray will have a list of index of each neighboring points
                % witin the all dataset (not within the minority subset)
                idx = obj.labelsAll(nnarray) == obj.minorityLabel;
                NofNonMinority = sum(~idx); % number of non-minority data
                obj.nnarrays{ii} = nnarray(idx); % keeps minority dataset only
                obj.weights(ii) = NofNonMinority/obj.NumNeighbors; % keeps the ratio of non-minority dataset
            end
        end

        function [T1, T2, featuresSubset] = createFeatureSubset(obj)
            featuresSubset = obj.featuresMinority;
            T1 =  size(obj.featuresMinority, 1);
            T2 = ceil(obj.num2Add*(obj.weights/sum(obj.weights))); % inner loop size array depend on outer loop num T1
        end

        function ret = validParams(obj)
            ret = validParams@SMOTE(obj);
            ret = ret && ~all(obj.weights==0);
        end        

    end
end
