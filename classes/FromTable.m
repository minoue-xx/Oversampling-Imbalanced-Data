classdef FromTable < handle
    properties
        oversamplingObj
        concatLabelData = false
    end

    methods
        function obj = FromTable(oversamplingObj)
            obj.oversamplingObj = oversamplingObj;
        end

        function newdata = run(obj, tbl, labelVarName)
            newdata = obj.predictImpl(tbl, labelVarName);
        end

        function newdata = predictImpl(obj, tbl, labelVarName)
            % TODO ラベルの列を特定しておき、concat時に元の位置に復元する
            X = removevars(tbl, labelVarName);
            xnames = string(X.Properties.VariableNames);
            X = X.Variables;

            y = tbl.(labelVarName);
            yname = labelVarName;
            
            newdata = obj.oversamplingObj.predictImpl(X, y);
            newdata = array2table(newdata, VariableNames=xnames);

            if obj.concatLabelData
                numNewdata = size(newdata,1);
                ydata = repmat(obj.oversamplingObj.minorityLabel, numNewdata,1);
                ydata = array2table(ydata, "VariableNames", yname);
                newdata = [newdata ydata]; % カラムの順序が入れ替わってしまう 
            end
        end

    end
end
