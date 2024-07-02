classdef SMOTE < handle
    properties
        NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 1
        minorityLabel (1,1) {double, string, categorical}
        num2Add (1,1) double {mustBePositive, mustBeInteger} = 1
        distance string
        concatNewData logical = false;
    end

    properties(SetAccess=protected)
        featuresMinority 
        featuresAll
        labelsAll
    end

    methods
        function obj = SMOTE(numNeighbors, num2Add, minorityLabel, distance)
            arguments
                numNeighbors = 5;
                num2Add = 10
                minorityLabel = 1;
                distance string {mustBeMember(distance, ...
                    ["cityblock", "chebychev", "correlation", "cosine", "euclidean", ...
                    "hamming", "jaccard", "mahalanobis", "minkowski", "seuclidean", "spearman"])} = "euclidean"
            end
            obj.NumNeighbors = numNeighbors;
            obj.distance = distance;
            obj.num2Add = num2Add;
            obj.minorityLabel = minorityLabel;
        end

        function newdata = run(obj, X, y)
            newdata = obj.predictImpl(X, y);
        end

        function newdata = predictImpl(obj, X, y)
            obj.featuresAll = X;
            obj.labelsAll = y;

            minorityIdx = y == obj.minorityLabel;
            obj.featuresMinority = X(minorityIdx, :);

            obj.calcAndSetParameters();

            if ~validParams(obj)
                newdata = [];
                warning("Oversampling is not executed. Return values will be empty.")
                return
            end

            [T1,T2,featuresSubset] = obj.createFeatureSubset();
            newdata = obj.createMinorFeatures(T1,T2,featuresSubset);

            if obj.concatNewData
                newdata = [X; newdata];
            end
        end

        function set.minorityLabel(obj, label)
            obj.minorityLabel = label;
        end

        function set.num2Add(obj, num)
            obj.num2Add = num;
        end
    end

    methods(Access=protected)
        function [T1, T2, featuresSubset] = createFeatureSubset(obj)
            NofMinorityData = size(obj.featuresMinority, 1);
            if NofMinorityData >= obj.num2Add
                idx = randperm(NofMinorityData, obj.num2Add);
                featuresSubset = obj.featuresMinority(idx, :);
                T1 = obj.num2Add;
                T2 = 1;
            else
                idx = randperm(NofMinorityData);
                featuresSubset = obj.featuresMinority(idx, :);
                T1 = NofMinorityData;
                T2 = ceil(obj.num2Add / NofMinorityData);
            end
        end

        function newFeatures = createMinorFeatures(obj, T1,T2, ...
                featuresSubset)
            newFeatures = zeros(obj.num2Add, size(obj.featuresMinority, 2));
            index = 1;
            for ii = 1:T1
                ySample = featuresSubset(ii, :);
                [nnarray, ~] = knnsearch(obj.featuresMinority, ySample, ...
                    'k', obj.NumNeighbors+1, 'SortIndices',true, 'Distance', obj.distance);
                nnarray = nnarray(2:end);

                if isscalar(T2)
                    innerLoopNum = T2;
                else
                    innerLoopNum = T2(ii); % depend on outer loop
                end
                for kk = 1:innerLoopNum
                    nn = datasample(nnarray, 1);
                    diff = obj.featuresMinority(nn, :) - ySample;
                    synthetic = ySample + obj.calcGain(ii,kk) .* diff;
                    newFeatures(index, :) = synthetic;
                    index = index + 1;

                    if index > obj.num2Add
                        break;
                    end
                end
            end
        end

        function ret = validParams(obj)
            ret = true;
            if isempty(obj.minorityLabel) || isempty(obj.num2Add)
                warning('minorityLabel and num2Add must be set before calling predictImpl.');
                ret = false;
            end

            if obj.num2Add <= 0
                ret = false;
            end
        end

        function calcAndSetParameters(obj)
            
        end

        function gain = calcGain(obj, ii, kk)
            gain = rand;
        end
    end
end

