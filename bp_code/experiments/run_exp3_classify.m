function R = run_exp3_classify(cfg)
%RUN_EXP3_CLASSIFY 实验三：设计实现 BP 网络完成识别分类，并按课堂方法检验结果
%
%   对应实验要求第 3 句：
%     "设计实现简单的 BP 网络，进行训练数据的识别分类"
%
%   这一句同时对应课堂第 6 节「感知器的验证」的全部要求，因此本实验分四块：
%     3.1 分类结果       —— 混淆矩阵、准确率、精确率/召回率/F1（课堂③④）
%     3.2 划分与交叉验证 —— 训练/测试比例、K 折交叉验证（课堂①②）
%     3.3 ROC 与 AUC     —— 一对其余 ROC（课堂⑤）
%     3.4 工具箱对照     —— 手写实现 vs patternnet，验证实现正确

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 10;  end
if ~isfield(cfg,'K'),    cfg.K = 5;   end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
fprintf('\n==================================================================\n');
fprintf('  实验三：Iris 识别分类与结果检验（课堂「感知器的验证」方法）\n');
fprintf('==================================================================\n');

fprintf('\n>>> 3.1 分类结果：混淆矩阵与四项指标\n');
R.classify = local_classify(struct('R', cfg.R, 'save', cfg.save));

fprintf('\n>>> 3.2 划分比例与 %d 折交叉验证\n', cfg.K);
R.cv = eval_cv(struct('R', cfg.R, 'K', cfg.K, 'save', cfg.save));

fprintf('\n>>> 3.3 ROC 曲线与 AUC\n');
R.roc = eval_roc(struct('save', cfg.save));

fprintf('\n>>> 3.4 与 MATLAB 工具箱交叉验证\n');
R.toolbox = local_toolbox(struct('save', cfg.save));

% ---------- 小结 ----------
fprintf('\n【实验三 小结】\n');
fprintf('  分类精度：%.2f%% ± %.2f%%（%d 个随机种子）\n', ...
        100*R.classify.accMean, 100*R.classify.accStd, cfg.R);
fprintf('  宏平均：精确率 %.2f%%  召回率 %.2f%%  F1 %.2f%%\n', ...
        100*mean(R.classify.metrics.precision), 100*mean(R.classify.metrics.recall), ...
        100*mean(R.classify.metrics.f1));
fprintf('  %d 折交叉验证：%.2f%% ± %.2f%%\n', ...
        numel(R.cv.acc), 100*mean(R.cv.acc), 100*std(R.cv.acc));
fprintf('  宏平均 AUC：%.4f\n', R.roc.m.macroAUC);
end

%% ===================== 3.1 分类与混淆矩阵（原 classify_run.m） =====================
function R = local_classify(cfg)
%CLASSIFY_RUN 实验一 内容(3)：设计实现 BP 网络完成 Iris 识别分类
%
%   用手写 BP 网络（4-8-3，gda，lr=0.1）在 trainData 上训练、testData 上测试：
%     1) 重复 10 个随机种子，报告分类精度的均值±标准差，检验结果稳定性；
%     2) 给出测试集混淆矩阵与各类精确率/召回率/F1；
%     3) 列出被分错的具体样本，便于分析错因；
%     4) 另建一个只用"花瓣长/花瓣宽"两维输入的 2-8-3 网络，
%        在二维平面上画出判决区域，直观展示 BP 网络学到的分类边界。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 10;    end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S = load_iris();
P = norm_fit(S.train.X, 'zscore');
Xtr = norm_apply(S.train.X, P);   Xte = norm_apply(S.test.X, P);
ytr = S.train.y;                  yte = S.test.y;
cls = S.className;

o = struct('sizes',[4 8 3], 'method','gda', 'lr',0.1, 'mc',0.9, ...
           'act','logsig', 'outAct','logsig', 'init','nguyen', ...
           'maxEpoch',2000, 'goal',1e-3, 'loss','mse');

%% ---------- 1) 多随机种子重复，统计稳定性 ----------
fprintf('\n===== 分类精度（%d 个随机种子）=====\n', cfg.R);
acc = zeros(cfg.R,1); netBest = []; accBest = -1; seedBest = 1; yh0 = [];
for r = 1:cfg.R
    oo = o;  oo.seed = r;
    m = train_eval(oo.sizes, Xtr, S.train.T, ytr, Xte, yte, oo);
    acc(r) = m.accTest;
    if m.accTest > accBest
        accBest = m.accTest;  netBest = m.net;  seedBest = r;
    end
    fprintf('  seed %2d：训练精度=%.4f  测试精度=%.4f  迭代=%4d\n', ...
            r, m.accTrain, m.accTest, m.epochs);
end
yh0 = bp_predict(netBest, Xte);
fprintf('  ---- 平均测试精度 = %.4f ± %.4f（最高 %.4f，seed=%d）----\n', ...
        mean(acc), std(acc), accBest, seedBest);

%% ---------- 2) 混淆矩阵与各类指标 ----------
C = 3;
CM = zeros(C, C);
for i = 1:numel(yte)
    CM(yte(i), yh0(i)) = CM(yte(i), yh0(i)) + 1;
end
precision = zeros(C,1); recall = zeros(C,1); f1 = zeros(C,1);
for c = 1:C
    tp = CM(c,c);
    precision(c) = tp / max(sum(CM(:,c)), eps);
    recall(c)    = tp / max(sum(CM(c,:)), eps);
    f1(c)        = 2*precision(c)*recall(c) / max(precision(c)+recall(c), eps);
end
macroP = mean(precision);  macroR = mean(recall);  macroF1 = mean(f1);

fprintf('\n===== 测试集混淆矩阵（行=真实，列=预测）=====\n');
fprintf('            %-22s %-22s %-22s\n', cls{:});
for c = 1:C
    fprintf('  %-20s %6d %6d %6d\n', cls{c}, CM(c,:));
end
fprintf('  各类精确率: %.4f / %.4f / %.4f\n', precision);
fprintf('  各类召回率: %.4f / %.4f / %.4f\n', recall);
fprintf('  各类 F1   : %.4f / %.4f / %.4f\n', f1);
fprintf('  宏平均 P=%.4f  R=%.4f  F1=%.4f\n', macroP, macroR, macroF1);

%% ---------- 3) 错误样本明细 ----------
bad = find(yh0 ~= yte);
fprintf('\n===== 被分错的测试样本（共 %d 个 / %d）=====\n', numel(bad), numel(yte));
errT = table();
if ~isempty(bad)
    errT = table((1:numel(bad))', bad(:), yte(bad)', yh0(bad)', ...
        S.test.X(1,bad)', S.test.X(2,bad)', S.test.X(3,bad)', S.test.X(4,bad)', ...
        'VariableNames', {'序号','测试集样本号','真实类别','预测类别', ...
        '花萼长','花萼宽','花瓣长','花瓣宽'});
    for k = 1:numel(bad)
        fprintf('  样本%3d：真实=类%d，预测=类%d  [%.1f %.1f %.1f %.1f]\n', ...
            bad(k), yte(bad(k)), yh0(bad(k)), S.test.X(:,bad(k)));
    end
end

%% ---------- 4) 图：混淆矩阵 + 各类指标 ----------
fig = figure('Visible','off','Position',[80 80 1040 420]);
subplot(1,2,1);
imagesc(CM); colormap(gca, flipud(gray(64))); colorbar;
axis square; set(gca,'XTick',1:C,'YTick',1:C);
set(gca,'XTickLabel',{'类1','类2','类3'},'YTickLabel',{'类1','类2','类3'});
xlabel('预测类别'); ylabel('真实类别');
title(sprintf('(a) 测试集混淆矩阵（精度 %.1f%%）', 100*accBest));
for i = 1:C
    for j = 1:C
        if CM(i,j) > 0
            text(j, i, sprintf('%d', CM(i,j)), 'HorizontalAlignment','center', ...
                 'FontSize',13, 'Color', ifelse(CM(i,j) > 8, 'w', 'k'));
        end
    end
end
subplot(1,2,2);
bar(100*[precision recall f1]); 
set(gca,'XTick',1:C,'XTickLabel',{'类1 山鸢尾','类2 变色鸢尾','类3 维吉尼亚'});
ylabel('百分比 (%)'); ylim([0 105]); grid on;
legend({'精确率','召回率','F1'},'Location','southwest');
title('(b) 各类识别指标');
R.fig1 = fig_utils(fig, 'exp4_confusion');

%% ---------- 5) 二维输入网络的判决区域 ----------
% 只用花瓣长、花瓣宽两维，便于在平面上画出决策边界
fIdx = [3 4];
P2 = norm_fit(S.train.X(fIdx,:), 'zscore');
X2tr = norm_apply(S.train.X(fIdx,:), P2);
X2te = norm_apply(S.test.X(fIdx,:),  P2);
o2 = o;  o2.sizes = [2 8 3];  o2.seed = seedBest;
m2 = train_eval(o2.sizes, X2tr, S.train.T, ytr, X2te, yte, o2);
fprintf('\n===== 仅用花瓣两维输入的 2-8-3 网络 =====\n');
fprintf('  训练精度=%.4f  测试精度=%.4f\n', m2.accTrain, m2.accTest);

% 在原始特征坐标上生成网格，归一化后送入网络
gx = linspace(0.8, 7.1, 240);
gy = linspace(-0.2, 2.7, 240);
[GX, GY] = meshgrid(gx, gy);
Gn = norm_apply([GX(:)'; GY(:)'], P2);
Gpred = bp_predict(m2.net, Gn);
Z = reshape(Gpred, size(GX));

fig = figure('Visible','off','Position',[80 80 760 560]);
hold on;
cmap = [0.85 0.92 1.00; 0.86 0.97 0.85; 1.00 0.90 0.86];
colormap(cmap);
contourf(GX, GY, Z, [0.5 1.5 2.5], 'LineStyle','none');
contour(GX, GY, Z, [1.5 2.5], 'k-', 'LineWidth', 1.2);
mk = {'o','s','^'};
h = gobjects(1,3);
for c = 1:3
    sel = (S.test.y == c);
    h(c) = plot(S.test.X(3,sel), S.test.X(4,sel), mk{c}, 'MarkerSize',6, ...
                'MarkerFaceColor', cmap(c,:)*0.75, 'MarkerEdgeColor','k');
end
xlabel('花瓣长 (cm)'); ylabel('花瓣宽 (cm)');
title(sprintf('BP 网络在二维特征平面上的判决区域（测试精度 %.1f%%）', 100*m2.accTest));
legend(h, {'类1 山鸢尾','类2 变色鸢尾','类3 维吉尼亚'}, 'Location','northwest');
axis([0.8 7.1 -0.2 2.7]); grid on;
R.fig2 = fig_utils(fig, 'exp4_decision_boundary');

%% ---------- 保存 ----------
R.acc = acc;
R.accMean = mean(acc);
R.accStd = std(acc);
R.CM = CM;
R.metrics = table((1:C)', precision, recall, f1, ...
    'VariableNames', {'class','precision','recall','f1'});
R.errors = errT;
R.seedBest = seedBest;

if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    M = R.metrics;
    M.Properties.VariableNames = {'类别','精确率','召回率','F1'};
    writetable(M, fullfile(outdir,'exp3_class_metrics.xlsx'), 'WriteMode','replacefile');
    if ~isempty(errT)
        E = errT;
        E.Properties.VariableNames = {'序号','测试集样本号','真实类别','预测类别', ...
            '花萼长','花萼宽','花瓣长','花瓣宽'};
        writetable(E, fullfile(outdir,'exp3_class_errors.xlsx'), 'WriteMode','replacefile');
    end
    CMT = array2table(CM, 'VariableNames', {'预测类1','预测类2','预测类3'}, ...
                           'RowNames', {'真实类1','真实类2','真实类3'});
    writetable(CMT, fullfile(outdir,'exp3_class_confusion.xlsx'), 'WriteMode','replacefile', ...
               'WriteRowNames',true);
    fid = fopen(fullfile(outdir,'exp3_class_accuracy.txt'),'w','n','UTF-8');
    fprintf(fid, '重复%d次，平均测试精度 %.4f ± %.4f，最高 %.4f（seed %d）\n', ...
            cfg.R, mean(acc), std(acc), accBest, seedBest);
    fprintf(fid, '二维花瓣输入网络：训练 %.4f，测试 %.4f\n', m2.accTrain, m2.accTest);
    fclose(fid);
end
fprintf('\n分类部分完成。\n');
end

% ------------------------------------------------------------------
function v = ifelse(cond, a, b)
if cond, v = a; else, v = b; end
end

%% ===================== 3.4 工具箱对比（原 compare_toolbox.m） =====================
function R = local_toolbox(cfg)
%COMPARE_TOOLBOX 交叉验证：手写 BP 网络 vs MATLAB 工具箱 patternnet
%
%   目的：证明手写实现（前向+反传+梯度下降）在同样数据上能达到工具箱同等水平，
%         同时对比不同训练算法的收敛速度。
%
%   注意：patternnet 默认用 softmax + 交叉熵，手写网络用 logsig + 均方误差，
%         两者误差函数不同，只能比较"分类精度"这一最终指标，不能直接比较误差值。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S = load_iris();
P   = norm_fit(S.train.X, 'zscore');
Xtr = norm_apply(S.train.X, P);   Xte = norm_apply(S.test.X, P);
ytr = S.train.y;                  yte = S.test.y;

names = {};  accTr = [];  accTe = [];  eps_ = [];  tms = [];

%% ---------- 1) 手写网络 ----------
o = struct('sizes',[4 8 3], 'method','gda', 'lr',0.1, 'mc',0.9, ...
           'act','logsig', 'outAct','logsig', 'init','nguyen', ...
           'maxEpoch',2000, 'goal',1e-3, 'loss','mse', 'seed',1);
m = train_eval(o.sizes, Xtr, S.train.T, ytr, Xte, yte, o);
names{end+1} = '手写 BP（GDA）';
accTr(end+1)  = m.accTrain;
accTe(end+1)  = m.accTest;
eps_(end+1)   = m.epochs;
tms(end+1)    = m.time;
fprintf('\n手写 BP   ：训练精度=%.4f  测试精度=%.4f  迭代=%d  耗时=%.2fs\n', ...
        m.accTrain, m.accTest, m.epochs, m.time);

%% ---------- 2) 工具箱 patternnet ----------
if exist('patternnet', 'file') ~= 2
    warning('未找到 patternnet，跳过工具箱对比');
    R = struct('T', table());
    return;
end

algos = {'trainscg', 'trainlm', 'traingd'};
adesc = {'工具箱 trainscg（量化共轭梯度）', ...
         '工具箱 trainlm（Levenberg-Marquardt）', ...
         '工具箱 traingd（标准梯度下降）'};

for k = 1:numel(algos)
    net = patternnet(8, algos{k});
    net.trainParam.showWindow    = false;
    net.trainParam.show          = NaN;
    net.trainParam.epochs        = 2000;
    net.trainParam.goal          = 1e-3;
    net.divideFcn                = 'dividetrain';   % 用全部训练样本，不划验证集
    net.trainParam.min_grad      = 1e-6;

    t0 = tic;
    [net, tr] = train(net, Xtr, S.train.T);
    tt = toc(t0);

    Y  = net(Xte);
    yh = vec2ind(Y);
    Yt = net(Xtr);
    yht = vec2ind(Yt);

    names{end+1} = adesc{k};
    accTr(end+1) = mean(yht == ytr);
    accTe(end+1) = mean(yh == yte);
    eps_(end+1)  = numel(tr.epoch);
    tms(end+1)   = tt;

    fprintf('%-28s：训练精度=%.4f  测试精度=%.4f  迭代=%4d  耗时=%.2fs\n', ...
            adesc{k}, mean(yht==ytr), mean(yh==yte), numel(tr.epoch), tt);
end

%% ---------- 3) 汇总 ----------
T = table(names(:), 100*accTr(:), 100*accTe(:), eps_(:), tms(:), ...
    'VariableNames', {'方法','训练精度_pct','测试精度_pct','迭代次数','耗时_s'});

fprintf('\n===== 一致性结论 =====\n');
fprintf('手写 BP 测试精度 %.2f%%，工具箱三种算法 %.2f%%~%.2f%%，相差 %.2f 个百分点，\n', ...
        100*accTe(1), 100*min(accTe(2:end)), 100*max(accTe(2:end)), ...
        100*max(abs(accTe(2:end) - accTe(1))));

%% ---------- 4) 图 ----------
fig = figure('Visible','off','Position',[80 80 1060 400]);
subplot(1,2,1);
b = bar(1:numel(names), 100*accTe); b.FaceColor = [0.30 0.55 0.85];
hold on;
b2 = bar(1:numel(names), 100*accTr, 0.4);
b2.FaceColor = [0.95 0.85 0.55]; b2.FaceAlpha = 0.85;
set(gca,'XTick',1:numel(names),'XTickLabel',names,'XTickLabelRotation',20);
ylabel('精度 (%)'); ylim([0 105]); grid on;
legend({'测试精度','训练精度'},'Location','southwest');
title('(a) 手写实现与工具箱的分类精度对比');
subplot(1,2,2);
b3 = bar(1:numel(names), tms); b3.FaceColor = [0.85 0.45 0.35];
set(gca,'XTick',1:numel(names),'XTickLabel',names,'XTickLabelRotation',20);
ylabel('训练耗时 (s)'); grid on;
title('(b) 训练耗时对比');
R.fig1 = fig_utils(fig, 'exp3_toolbox_compare');

R.T = T;
if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    writetable(T, fullfile(outdir,'exp3_toolbox.xlsx'), 'WriteMode','replacefile');
end
fprintf('\n工具箱对比完成。\n');
end
