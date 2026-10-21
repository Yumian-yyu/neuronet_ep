function [fpr, tpr, auc] = roc_auc(score, truth)
%ROC_AUC 二分类 ROC 曲线与 AUC（一对其余法用）
%   score : 1xN 分数，越大越倾向于正类
%   truth : 1xN 逻辑/0-1 真实标记
%   auc   : ROC 曲线下面积，用梯形法积分

score = double(score(:));
truth = logical(truth(:));

P = sum(truth);          % 正类个数
N = sum(~truth);         % 负类个数
if P == 0 || N == 0
    fpr = [0 1]; tpr = [0 1]; auc = NaN; return;
end

[s, idx] = sort(score, 'descend');   %#ok<ASGLU>
t = truth(idx);

tpr = cumsum(t)  / P;
fpr = cumsum(~t) / N;
fpr = [0; fpr];  tpr = [0; tpr];

auc = trapz(fpr, tpr);
end
