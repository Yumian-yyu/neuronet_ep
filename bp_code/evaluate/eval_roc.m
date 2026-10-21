function R = eval_roc(cfg)
%EVAL_ROC 课堂「感知器的验证」⑤：ROC 曲线与 AUC
%
%   三分类按「一对其余」拆成三个二分类问题：以第 c 类的输出值作为
%   "属于第 c 类"的置信分数，其余两类合并为负类，逐个移动判决阈值
%   得到 ROC 曲线；曲线下面积即 AUC，0.5 相当于随机猜测、1 表示完全可分。
%
%   AUC 反映全阈值范围内的排序质量，比只看 0.5 阈值的准确率信息量更大。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S = load_iris();
P   = norm_fit(S.train.X, 'zscore');
Xtr = norm_apply(S.train.X, P);
Xte = norm_apply(S.test.X, P);

net = bp_init([4 8 3], 'seed', 1);
net = bp_train(net, Xtr, S.train.T, 'method','gda', 'lr',0.1, 'mc',0.9, ...
               'maxEpoch',2000, 'goal',1e-3);

m = eval_classifier(net, Xte, S.test.y);
fprintf('\n===== 测试集各类别 AUC =====\n');
for c = 1:3
    fprintf('  类%d %-18s AUC = %.4f\n', c, S.className{c}, m.AUC(c));
end
fprintf('  宏平均 AUC = %.4f\n', m.macroAUC);
fprintf('  （对照：准确率 = %.2f%%，宏平均 F1 = %.2f%%）\n', 100*m.acc, 100*m.macroF1);

A = bp_forward(net, Xte);  Y = A{end};
fig = figure('Visible','off','Position',[80 80 620 520]); hold on;
for c = 1:3
    [f, t, ~] = roc_auc(Y(c,:), S.test.y==c);
    plot(f, t, 'LineWidth', 1.8);
end
plot([0 1], [0 1], 'k--');
grid on; xlabel('假正例率 FPR'); ylabel('真正例率 TPR');
title(sprintf('一对其余 ROC 曲线（宏平均 AUC = %.4f）', m.macroAUC));
legend([S.className, {'随机猜测'}], 'Location','southeast');
axis([0 1 0 1]);
R.fig = fig_utils(fig, 'exp3_roc_figure');

R.m = m;
if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    Ta = table((1:3)', S.className(:), m.AUC(:), m.P(:), m.R(:), m.F1(:), ...
        'VariableNames', {'class','name','auc','precision','recall','f1'});
    Ta.Properties.VariableNames = {'类别','名称','AUC','精确率','召回率','F1'};
    writetable(Ta, fullfile(outdir,'exp3_roc_auc.xlsx'), 'WriteMode','replacefile');
end
fprintf('\nROC/AUC 完成。\n');
end
