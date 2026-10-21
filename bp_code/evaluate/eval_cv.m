function R = eval_cv(cfg)
%EVAL_CV 课堂「感知器的验证」①②：数据集划分与 K 折交叉验证
%
%   ① 划分比例：训练集应占 70%~80%、测试集 20%~30%。
%      在 totalData 上分层随机划分，对比 80%/20% 与 50%/50%，
%      每种比例重抽 10 次、每次重新训练，看划分比例对精度的影响。
%   ② K 折交叉验证：分 K 个互斥子集轮流验证、取平均，
%      避免单次划分的偶然误差。
%
%   每个结果都用 eval_classifier 统一评价（准确率、宏平均 F1、宏平均 AUC）。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 10;  end
if ~isfield(cfg,'K'),    cfg.K = 5;   end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S = load_iris();
X = S.total.X;  y = S.total.y(:);  T = S.total.T;
base = struct('sizes',[4 8 3], 'method','gda', 'lr',0.1, 'mc',0.9, ...
              'maxEpoch',2000, 'goal',1e-3, 'act','logsig', 'init','nguyen');

%% ============ 一、训练/测试划分比例 ============
fprintf('\n===== ① 训练/测试划分比例（每种 %d 次分层随机划分）=====\n', cfg.R);
rng(7);
fracs  = [0.8 0.5];
accS   = zeros(numel(fracs), cfg.R);
f1S    = zeros(numel(fracs), cfg.R);
aucS   = zeros(numel(fracs), cfg.R);

for i = 1:numel(fracs)
    fr = fracs(i);
    for r = 1:cfg.R
        itr = [];  ite = [];
        for c = 1:3
            idx = find(y==c);  idx = idx(randperm(numel(idx)));
            ntr = round(fr*numel(idx));
            itr = [itr; idx(1:ntr)];  ite = [ite; idx(ntr+1:end)]; %#ok<AGROW>
        end
        Pk = norm_fit(X(:,itr), 'zscore');
        o = base;  o.seed = r;
        net = bp_init(o.sizes, 'seed',o.seed, 'act',o.act, 'init',o.init);
        net = bp_train(net, norm_apply(X(:,itr),Pk), T(:,itr), 'method',o.method, ...
                       'lr',o.lr, 'mc',o.mc, 'maxEpoch',o.maxEpoch, 'goal',o.goal);
        m = eval_classifier(net, norm_apply(X(:,ite),Pk), y(ite));
        accS(i,r) = m.acc;  f1S(i,r) = m.macroF1;  aucS(i,r) = m.macroAUC;
    end
    fprintf('  训练%3d / 测试%3d ：精度 %.4f ± %.4f   F1 %.4f    AUC %.4f\n', ...
            round(fr*150), 150-round(fr*150), mean(accS(i,:)), std(accS(i,:)), ...
            mean(f1S(i,:)), mean(aucS(i,:)));
end

%% ============ 二、K 折交叉验证 ============
fprintf('\n===== ② %d 折交叉验证（totalData 150 条）=====\n', cfg.K);
rng(2024);
cv  = cvpartition(numel(y), 'KFold', cfg.K);
acc = zeros(cfg.K,1);  f1 = zeros(cfg.K,1);  auc = zeros(cfg.K,1);  nTe = zeros(cfg.K,1);

for k = 1:cfg.K
    itr = training(cv,k);  ite = test(cv,k);
    Pk = norm_fit(X(:,itr), 'zscore');
    o = base;  o.seed = k;
    net = bp_init(o.sizes, 'seed',o.seed, 'act',o.act, 'init',o.init);
    net = bp_train(net, norm_apply(X(:,itr),Pk), T(:,itr), 'method',o.method, ...
                   'lr',o.lr, 'mc',o.mc, 'maxEpoch',o.maxEpoch, 'goal',o.goal);
    m = eval_classifier(net, norm_apply(X(:,ite),Pk), y(ite));
    acc(k) = m.acc;  f1(k) = m.macroF1;  auc(k) = m.macroAUC;  nTe(k) = sum(ite);
    fprintf('  第%d折：测试%2d条  精度 %.4f   F1 %.4f   AUC %.4f\n', ...
            k, nTe(k), acc(k), f1(k), auc(k));
end
fprintf('  ---- 平均：精度 %.4f ± %.4f   F1 %.4f   AUC %.4f ----\n', ...
        mean(acc), std(acc), mean(f1), mean(auc));

%% ============ 三、作图 ============
fig = figure('Visible','off','Position',[80 80 1040 410]);

subplot(1,2,1);
yy = [100*mean(accS,2), 100*mean(f1S,2), 100*mean(aucS,2)];
b = bar(yy); hold on;
b(1).FaceColor = [0.30 0.55 0.85];
b(2).FaceColor = [0.90 0.55 0.30];
b(3).FaceColor = [0.45 0.75 0.45];
set(gca,'XTick',1:2,'XTickLabel',{'训练120/测试30','训练75/测试75'});
ylabel('百分比 (%)'); ylim([0 105]); grid on;
legend({'准确率','宏平均 F1','宏平均 AUC'},'Location','southwest');
title(sprintf('(a) 划分比例的影响（各重抽 %d 次）', cfg.R));

subplot(1,2,2);
bar(1:cfg.K, 100*acc); hold on;
plot([0 cfg.K+1], 100*[mean(acc) mean(acc)], 'r--', 'LineWidth',1.5);
grid on; ylim([85 102]);
set(gca,'XTick',1:cfg.K,'XTickLabel',arrayfun(@(k) sprintf('第%d折',k), 1:cfg.K, ...
    'UniformOutput',false));
ylabel('测试精度 (%)');
title(sprintf('(b) %d 折交叉验证：平均 %.2f%% ± %.2f%%', ...
      cfg.K, 100*mean(acc), 100*std(acc)));
R.fig = fig_utils(fig, 'exp3_cv_figure');

%% ============ 四、保存 ============
R.acc = acc;  R.f1 = f1;  R.auc = auc;
R.splitAcc = accS;  R.splitF1 = f1S;  R.splitAUC = aucS;

if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end

    Tc = table((1:cfg.K)', nTe, acc, f1, auc, ...
        'VariableNames', {'fold','nTest','acc','f1','auc'});
    Tc(end+1,:) = {NaN, sum(nTe), mean(acc), mean(f1), mean(auc)};
    Tc.Properties.VariableNames = {'折次','测试样本数','测试精度','宏平均F1','宏平均AUC'};
    writetable(Tc, fullfile(outdir,'exp3_cv.xlsx'), 'WriteMode','replacefile');

    Ts = table(round(fracs(:)*150), 150-round(fracs(:)*150), ...
        mean(accS,2), std(accS,0,2), mean(f1S,2), mean(aucS,2), ...
        'VariableNames', {'nTrain','nTest','accMean','accStd','f1Mean','aucMean'});
    Ts.Properties.VariableNames = {'训练样本数','测试样本数','平均精度','精度标准差', ...
        '平均F1','平均AUC'};
    writetable(Ts, fullfile(outdir,'exp3_cv_split.xlsx'), 'WriteMode','replacefile');
end
fprintf('\n划分与交叉验证完成。\n');
end
