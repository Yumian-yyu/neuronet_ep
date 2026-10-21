function m = eval_classifier(net, X, y)
%EVAL_CLASSIFIER 按课堂「感知器的验证」一节的方法评价一个分类器
%
%   m = eval_classifier(net, X, y)
%
%   一次性给出课堂要求的全部指标：
%     混淆矩阵 CM              （真实 × 预测）
%     准确率  acc              = 对角线之和 / 总数
%     精确率  P、召回率 R、F1   （逐类 + 宏平均）
%     ROC/AUC                  （一对其余，逐类 + 宏平均）
%
%   这样所有实验脚本都调用同一个评价函数，指标口径完全一致，结果才可比。

yh = bp_predict(net, X);   yh = yh(:);
y  = y(:);
C  = net.sizes(end);

% ---- 混淆矩阵 ----
CM = zeros(C, C);
for i = 1:numel(y)
    CM(y(i), yh(i)) = CM(y(i), yh(i)) + 1;
end

% ---- 准确率 / 精确率 / 召回率 / F1 ----
P  = nan(C,1);  R = nan(C,1);  F1 = nan(C,1);
for c = 1:C
    tp = CM(c,c);
    denP = sum(CM(:,c));
    denR = sum(CM(c,:));
    if denP > 0, P(c) = tp / denP; end
    if denR > 0, R(c) = tp / denR; end
    if denP > 0 && denR > 0 && (P(c) + R(c)) > 0
        F1(c) = 2*P(c)*R(c) / (P(c) + R(c));
    end
end

% ---- ROC / AUC（一对其余）----
A = bp_forward(net, X);  Y = A{end};
AUC = nan(C,1);
for c = 1:C
    if any(y==c) && any(y~=c)
        [~, ~, a] = roc_auc(Y(c,:), y==c);
        AUC(c) = a;
    end
end

m.CM       = CM;
m.acc      = mean(yh == y);
m.P        = P;
m.R        = R;
m.F1       = F1;
m.macroP   = mean(P,  'omitnan');
m.macroR   = mean(R,  'omitnan');
m.macroF1  = mean(F1, 'omitnan');
m.AUC      = AUC;
m.macroAUC = mean(AUC,'omitnan');
m.yh       = yh;
end
