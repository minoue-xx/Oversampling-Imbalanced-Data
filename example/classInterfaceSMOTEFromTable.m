clear;
close all;
addpath('./classes');
rng(0)

Ndata = 2000;
data = rand(Ndata,2);

x = data(:,1);
y = data(:,2);

% center = (0.4,0.4), radius = 0.3
idx1a = (x-0.4).^2 + (y-0.4).^2 < 0.3^2;
% center = (0.8,0.8), radius = 0.05
idx1b = (x-0.8).^2 + (y-0.8).^2 < 0.05^2;
% center = (0.9,0.1), radius = 0.1
idx2 = (x-0.9).^2 + (y-0.1).^2 < 0.1^2;

% decrease the number of samples for two minority class
undersampleRate = 4; % Undersample rate
data1 = data(idx1a|idx1b,:);
data1 = data1(1:undersampleRate:end,:);
data2 = data(idx2,:);
data2 = data2(1:undersampleRate:end,:);

% delete those from the original datset
data(idx1a|idx1b|idx2,:) = [];

label0 = repmat("class0",length(data),1);
label1 = repmat("class1",length(data1),1);
label2 = repmat("class2",length(data2),1);

dataset = array2table([data;data1;data2]);
dataset = addvars(dataset, [label0;label1;label2],...
    'NewVariableNames','label');
labels = dataset(:,end);
t = tabulate(dataset.label)

uniqueLabels = string(t(:,1));
labelCounts = cell2mat(t(:,2));

num2AddList = [0,200,20];

newdata = table;


K = 10;
for ii = 1:length(num2AddList)
    % マイナーラベル
    minorityLabel = uniqueLabels(ii);
    num2Add = num2AddList(ii);
    if num2Add == 0
        continue
    end

    % SMOTEインスタンスの作成
    % smote = FromTable(SMOTE(K,num2Add,minorityLabel,"seuclidean"));
    % smote = FromTable(SafeLevelSMOTE(K,num2Add,minorityLabel,"seuclidean"));
    % smote = FromTable(BorderlineSMOTE(K,num2Add,minorityLabel,"seuclidean"));
    smote = FromTable(ADASYN(K,num2Add,minorityLabel,"seuclidean"));

    % 学習とサンプリング同時実施
    smote.concatLabelData = true;
    tmp = smote.run(dataset, "label");
    newdata = [newdata;tmp];
end

% % 新しいデータのプロット
figure;
gscatter(dataset.Var1,dataset.Var2,dataset.label,'krr','oo^',4,'off');
hold on
h = gscatter(newdata.Var1,newdata.Var2,newdata.label,'bb','o^',5,'off');
for n = 1:length(h)
    color = get(h(n),'Color');
    set(h(n), 'MarkerFaceColor', color);
end
hold off