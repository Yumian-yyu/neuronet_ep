function R = run_exp1_network(cfg)
%RUN_EXP1_NETWORK 实验一：BP 网络拓扑结构与各项参数对训练结果的影响
%
%   对应实验要求第 1 句：
%     "观察比较 BP 网络拓扑结构及其它各项参数变化对于训练结果的影响"
%
%   分两部分：
%     1.1 拓扑结构 —— 隐层数、隐层神经元个数（16 种结构，两种输入条件）
%     1.2 各项参数 —— 学习率、动量、激活函数、初始化、训练算法、输出层、稳定性
%
%   R.topology / R.params 分别是两部分的完整结果结构体。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 5;   end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
fprintf('\n==================================================================\n');
fprintf('  实验一：拓扑结构与各项参数对训练结果的影响\n');
fprintf('==================================================================\n');

% ---------- 1.1 拓扑结构 ----------
fprintf('\n>>> 1.1 拓扑结构（隐层数 / 隐层神经元个数）\n');
R.topology = local_topology(struct('R', cfg.R, 'save', cfg.save));

% ---------- 1.2 各项参数 ----------
fprintf('\n>>> 1.2 各项参数（学习率 / 动量 / 激活 / 初始化 / 算法 / 输出层）\n');
R.params = local_params(struct('R', cfg.R, 'save', cfg.save));

% ---------- 小结 ----------
T = R.topology.T;
Tz = T(T.norm == "zscore", :);
s1 = Tz(Tz.nHL == 1, :);
s2 = Tz(Tz.nHL == 2, :);
fprintf('\n【实验一 小结】\n');
fprintf('  单隐层：达标迭代 %.0f~%.0f 轮，测试精度 %.1f%%~%.1f%%\n', ...
        min(s1.epochs), max(s1.epochs), 100*min(s1.accTest), 100*max(s1.accTest));
fprintf('  双隐层：达标迭代 %.0f~%.0f 轮，测试精度 %.1f%%~%.1f%%\n', ...
        min(s2.epochs), max(s2.epochs), 100*min(s2.accTest), 100*max(s2.accTest));
fprintf('  → 拓扑结构主要影响收敛速度，对最终精度影响很小\n');

Tp = R.params.T;
g1 = Tp(Tp.group == "学习率", :);
g5 = Tp(Tp.group == "权值初始化", :);
gz = g5(g5.label == "全零", :);
fprintf('  学习率 0.01 → 2.00：精度 %.1f%% → %.1f%%\n', ...
        100*g1.accTest(1), 100*g1.accTest(end));
fprintf('  全零初始化：精度 %.1f%%（对称性无法打破）\n', 100*gz.accTest(1));
end

%% ===================== 1.1 拓扑结构（原 network_topology.m） =====================
function R = local_topology(cfg)
%NETWORK_TOPOLOGY 实验一 内容(1)：BP 网络拓扑结构对训练结果的影响
%
%   控制变量：训练算法(gda, lr=0.1)、初始化(Nguyen-Widrow)、随机种子序列、
%   迭代上限都固定，只改变"隐层数"和"隐层神经元个数"。
%   每个结构用 5 个不同随机种子重复，报告均值±标准差。
%
%   同时设置两种输入条件，用来说明"拓扑的影响"与"数据预处理"的耦合：
%     zscore —— 标准化输入（推荐做法）
%     none   —— 原始量纲直接输入
%
%   两种评价方式：
%     (A) 固定 500 轮预算：mse500 / acc500（反映学习速度与容量）
%     (B) 训练到 MSE<=1e-3 或 2000 轮：epochs（收敛速度）、accTe（最终泛化）
%
%   R.T —— 结果表格；R.fig —— 图片路径

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),        cfg.R = 5;          end
if ~isfield(cfg,'maxEpoch'), cfg.maxEpoch = 2000; end
if ~isfield(cfg,'budget'),   cfg.budget = 500;    end
if ~isfield(cfg,'save'),     cfg.save = true;     end

fig_utils();
S = load_iris();
yte = S.test.y;
ytr = S.train.y;

% ---------- 待比较的拓扑结构 ----------
singleH = [3 4 5 6 8 10 12 15 20 30 50];
multiH  = {[8 4], [10 6], [15 8], [20 10], [30 15]};

cfgList = cell(0,1);
for h = singleH
    cfgList{end+1} = struct('name', sprintf('4-%d-3', h), 'sizes', [4 h 3], ...
                            'nHL', 1, 'h1', h, 'h2', NaN);
end
for k = 1:numel(multiH)
    v = multiH{k};
    cfgList{end+1} = struct('name', sprintf('4-%d-%d-3', v(1), v(2)), ...
                            'sizes', [4 v 3], 'nHL', 2, 'h1', v(1), 'h2', v(2));
end

nC    = numel(cfgList);
nR    = cfg.R;
norms = {'zscore', 'none'};
zname = {'标准化输入', '原始输入'};
base  = struct('method','gda','lr',0.1,'mc',0.9, ...
               'act','logsig','outAct','logsig','init','nguyen');

rows   = cell(nC*2, 1);
curves = cell(1,0);  curveNames = cell(1,0);
rowIdx = 0;

for s = 1:numel(norms)
    P   = norm_fit(S.train.X, norms{s});
    Xtr = norm_apply(S.train.X, P);
    Xte = norm_apply(S.test.X, P);

    fprintf('\n----- 输入条件：%s -----\n', zname{s});
    fprintf('%-11s %7s %9s %10s %9s %9s\n', ...
            '结构','参数数','精度@500轮','MSE@500轮','达标迭代','测试精度');

    for i = 1:nC
        g = cfgList{i};
        accB = zeros(nR,1); mseB = zeros(nR,1);
        accT = zeros(nR,1); accR = zeros(nR,1);
        ep   = zeros(nR,1); hit  = false(nR,1); nPar = 0;

        for r = 1:nR
            % (A) 固定预算
            oA = base;  oA.seed = r;  oA.maxEpoch = cfg.budget;  oA.goal = 0;
            mA = train_eval(g.sizes, Xtr, S.train.T, ytr, Xte, yte, oA);
            accB(r) = mA.accTest;  mseB(r) = mA.mse;

            % (B) 训练到目标误差
            oB = base;  oB.seed = r;  oB.maxEpoch = cfg.maxEpoch;  oB.goal = 1e-3;
            mB = train_eval(g.sizes, Xtr, S.train.T, ytr, Xte, yte, oB);
            accT(r) = mB.accTest;  accR(r) = mB.accTrain;
            ep(r)   = mB.epochs;   hit(r)  = strcmp(mB.stop,'goal');
            nPar    = mB.nParam;
            if r == 1 && s == 1
                curves{end+1} = mB.perf(:)';  curveNames{end+1} = g.name;
            end
        end

        rowIdx = rowIdx + 1;
        rows{rowIdx} = table({g.name}, {norms{s}}, g.nHL, g.h1, g.h2, nPar, ...
            mean(mseB), mean(accB), std(accB), mean(ep), std(ep), mean(hit), ...
            mean(accT), std(accT), mean(accR), mean(accR)-mean(accT), ...
            'VariableNames', {'topo','norm','nHL','h1','h2','nParam', ...
            'mse500','acc500','acc500std','epochs','epochsStd','hitRate', ...
            'accTest','accTestStd','accTrain','gap'});

        fprintf('%-11s %7d %9.4f %10.2e %9.0f %9.4f\n', g.name, nPar, ...
                mean(accB), mean(mseB), mean(ep), mean(accT));
    end
end

T = vertcat(rows{:});

% ---------- 图 1：两种输入条件下的测试精度 ----------
fig = figure('Visible','off','Position',[80 80 1060 400]);
for s = 1:2
    M = T(strcmp(T.norm, norms{s}), :);
    isS = M.nHL == 1;
    subplot(1,2,s);
    errorbar(M.h1(isS), M.acc500(isS), M.acc500std(isS), '-o', ...
             'MarkerFaceColor','b','CapSize',4); hold on;
    plot(M.h1(~isS), M.acc500(~isS), 's','MarkerSize',7, ...
         'MarkerFaceColor',[0.85 0.33 0.10], 'LineStyle','none');
    grid on; xlabel('第一隐层神经元个数'); ylabel('测试集分类精度（500 轮预算）');
    title(sprintf('(%s) %s', char('a'+s-1), zname{s}));
    legend({'单隐层','双隐层'}, 'Location','best'); ylim([0.87 1.02]);
end
R.fig1 = fig_utils(fig, 'exp1_topology_accuracy');

% ---------- 图 2：收敛速度 + 500 轮后的拟合精度 ----------
fig = figure('Visible','off','Position',[80 80 1060 400]);

subplot(1,2,1);
M  = T(strcmp(T.norm,'zscore'), :);
isS = M.nHL == 1;
plot(M.h1(isS), M.epochs(isS), '-o','MarkerFaceColor','b'); hold on;
plot(M.h1(~isS), M.epochs(~isS), 's','MarkerSize',7, ...
     'MarkerFaceColor',[0.85 0.33 0.10], 'LineStyle','none');
grid on; xlabel('第一隐层神经元个数');
ylabel('达到 MSE=10^{-3} 的迭代次数');
title('(a) 收敛速度（标准化输入）');
legend({'单隐层','双隐层'}, 'Location','best');

subplot(1,2,2);
cols = {[0.00 0.45 0.74], [0.85 0.33 0.10]};
for s = 1:2
    M = T(strcmp(T.norm, norms{s}), :);
    M = M(M.nHL == 1, :);                 % 为清晰起见只画单隐层链
    semilogy(M.h1, M.mse500, '-o', 'Color',cols{s}, 'MarkerFaceColor',cols{s});
    hold on;
end
grid on; xlabel('第一隐层神经元个数'); ylabel('500 轮后的训练 MSE');
title('(b) 相同时代预算下的拟合程度');
legend({'标准化输入','原始输入'}, 'Location','best');
R.fig2 = fig_utils(fig, 'exp1_topology_epochs');

% ---------- 图 3：典型结构的误差下降曲线（标准化输入） ----------
pick = {'4-3-3','4-8-3','4-20-3','4-50-3','4-10-6-3'};
fig = figure('Visible','off','Position',[80 80 720 430]);
hold on;
sty = {'-','-','-','-','--'};
for k = 1:numel(pick)
    idx = find(strcmp(curveNames, pick{k}), 1);
    if isempty(idx), continue; end
    y = curves{idx};
    semilogy(1:numel(y), y, sty{min(k,numel(sty))}, 'LineWidth',1.6);
end
grid on; xlabel('迭代次数'); ylabel('训练均方误差 MSE');
set(gca,'YScale','log'); ylim([1e-4 1]);
yline(1e-3, 'k:', '目标误差 10^{-3}', 'LabelHorizontalAlignment','left');
title('不同拓扑结构的误差下降曲线（标准化输入）');
legend(pick, 'Location','northeast');
R.fig3 = fig_utils(fig, 'exp1_curves');

% ---------- 图 4：参数规模 vs 测试精度（两种输入条件） ----------
fig = figure('Visible','off','Position',[80 80 700 450]);
hold on; grid on;
h1s = T(strcmp(T.norm,'zscore'), :);
h2s = T(strcmp(T.norm,'none'), :);
h(1) = plot(h1s.nParam, 100*h1s.acc500, 'o', 'MarkerFaceColor','b', 'MarkerSize',8);
h(2) = plot(h2s.nParam, 100*h2s.acc500, 's', 'MarkerFaceColor',[0.85 0.33 0.10], ...
            'MarkerSize',8);
ylabel('测试集分类精度 (%)'); xlabel('网络可训练参数个数');
title('参数规模与泛化精度的关系');
legend(h, zname, 'Location','southeast'); ylim([88 98]);
R.fig4 = fig_utils(fig, 'exp1_topology_params');

R.T = T;
if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    Tzh = T;
    Tzh.Properties.VariableNames = {'拓扑结构','输入条件','隐层数','第一隐层节点', ...
        '第二隐层节点','参数个数','MSE_500轮','精度_500轮','精度_500轮_标准差', ...
        '达标迭代次数','达标迭代_标准差','达标率','测试精度','测试精度_标准差', ...
        '训练精度','泛化间隙'};
    writetable(Tzh, fullfile(outdir,'exp1_topology.xlsx'), 'WriteMode','replacefile');
end
fprintf('\n拓扑结构部分完成。\n');
end

%% ===================== 1.2 各项参数（原 network_params.m） =====================
function R = local_params(cfg)
%NETWORK_PARAMS 实验一 内容(1)：BP 网络各项参数变化对训练结果的影响
%
%   基准网络 4-8-3，标准化输入，每项参数单独变化（其余保持不变），
%   每个取值用 5 个随机种子重复，统计训练误差与测试精度。
%
%   考察参数：
%     1) 学习率 lr        （标准梯度下降 gd，观察收敛快慢与振荡/发散）
%     2) 动量因子 mc      （带动量梯度下降 gdm）
%     3) 隐层激活函数     （logsig / tansig / relu）
%     4) 权值初始化方式   （Nguyen-Widrow / 均匀分布大小 / 正态 / 全零）
%     5) 训练算法         （gd / gdm / gda / gdx）
%     6) 输出层与误差函数 （logsig+MSE vs softmax+交叉熵）
%
%   R.T —— 全部结果汇总表；R.fig —— 图片路径

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 5;    end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S   = load_iris();
P   = norm_fit(S.train.X, 'zscore');
D.Xtr = norm_apply(S.train.X, P);  D.Ttr = S.train.T;  D.ytr = S.train.y;
D.Xte = norm_apply(S.test.X,  P);  D.yte = S.test.y;

base = struct('sizes',[4 8 3], 'method','gda', 'lr',0.1, 'mc',0.9, ...
              'act','logsig', 'outAct','logsig', 'init','nguyen', ...
              'maxEpoch',500, 'goal',0, 'loss','mse');
Rr = cfg.R;
groups = {};  grpName = {};

% ---------- 1) 学习率 ----------
lrList = [0.01 0.05 0.1 0.3 0.5 1.0 2.0];
v = cell(1,numel(lrList));
for k = 1:numel(lrList)
    v{k} = struct('label', sprintf('lr=%.2f', lrList(k)), 'lr', lrList(k), 'method','gd');
end
groups{end+1} = v;  grpName{end+1} = '学习率';

% ---------- 2) 动量因子 ----------
mcList = [0 0.5 0.8 0.9 0.95];
v = cell(1,numel(mcList));
for k = 1:numel(mcList)
    v{k} = struct('label', sprintf('mc=%.2f', mcList(k)), 'mc', mcList(k), 'method','gdm');
end
groups{end+1} = v;  grpName{end+1} = '动量因子';

% ---------- 3) 隐层激活函数 ----------
acts = {'logsig','tansig','relu'};
v = cell(1,numel(acts));
for k = 1:numel(acts)
    v{k} = struct('label', acts{k}, 'act', acts{k});
end
groups{end+1} = v;  grpName{end+1} = '隐层激活函数';

% ---------- 4) 权值初始化 ----------
v = { struct('label','Nguyen-Widrow','init','nguyen'), ...
      struct('label','均匀[-0.5,0.5]','init','uniform','wRange',0.5), ...
      struct('label','均匀[-3,3]','init','uniform','wRange',3), ...
      struct('label','正态 randn','init','randn'), ...
      struct('label','全零','init','zeros') };
groups{end+1} = v;  grpName{end+1} = '权值初始化';

% ---------- 5) 训练算法 ----------
algs = {'gd','gdm','gda','gdx'};
v = cell(1,numel(algs));
for k = 1:numel(algs)
    v{k} = struct('label', upper(algs{k}), 'method', algs{k});
end
groups{end+1} = v;  grpName{end+1} = '训练算法';

% ---------- 6) 输出层与误差函数 ----------
v = { struct('label','logsig + MSE','outAct','logsig','loss','mse'), ...
      struct('label','softmax + 交叉熵','outAct','softmax','loss','ce') };
groups{end+1} = v;  grpName{end+1} = '输出层与误差函数';

% ---------- 7) 线性输出层下的学习率上限 ----------
% 输出层为 logsig 时误差项被限制在 0.25 以内、等效步长有界，学习率不易发散；
% 改成 purelin(线性) 输出后误差曲面曲率变大，学习率超过临界值就会振荡发散。
v = { struct('label','purelin  lr=1','outAct','purelin','method','gd','lr',1,  'maxEpoch',300), ...
      struct('label','purelin  lr=2','outAct','purelin','method','gd','lr',2,  'maxEpoch',300), ...
      struct('label','purelin  lr=5','outAct','purelin','method','gd','lr',5,  'maxEpoch',300) };
groups{end+1} = v;  grpName{end+1} = '线性输出层的稳定性';

% ================= 逐组执行 =================
allT = cell(1,numel(groups));
figs = cell(1,0);

for g = 1:numel(groups)
    fprintf('\n===== 参数组 %d/%d：%s =====\n', g, numel(groups), grpName{g});
    [T, curves, labels] = local_sweep(base, groups{g}, D, Rr);
    T.group = repmat({grpName{g}}, height(T), 1);
    allT{g} = T;

    for k = 1:height(T)
        fprintf('  %-18s  MSE=%.3e  训练精度=%.4f  测试精度=%.4f±%.4f  迭代=%4.0f\n', ...
            labels{k}, T.mse(k), T.accTrain(k), T.accTest(k), T.accTestStd(k), T.epochs(k));
    end

    % ---- 每组一张图：(a) 误差曲线  (b) 终值与精度 ----
    fig = figure('Visible','off','Position',[80 80 1080 420]);
    subplot(1,2,1);
    hold on;
    cols = lines(numel(curves));
    for k = 1:numel(curves)
        y = curves{k};
        if isempty(y) || ~any(isfinite(y)), continue; end
        y(y > 1e3) = NaN;      % 发散点置空，避免把坐标轴拉爆
        semilogy(1:numel(y), y, 'Color', cols(k,:), 'LineWidth',1.5);
    end
    set(gca,'YScale','log');
    if g == 7, ylim([1e-3 1e2]); end
    grid on; xlabel('迭代次数'); ylabel('训练均方误差 MSE（对数轴）');
    title(sprintf('(a) %s：误差下降曲线', grpName{g}));
    legend(labels, 'Location','northeast', 'Interpreter','none');

    subplot(1,2,2);
    if g == 7
        % 发散组的 MSE 为 Inf，柱状图无意义，只画分类精度
        b = bar(1:height(T), 100*T.accTest); b.FaceColor = [0.30 0.55 0.85];
        hold on;
        errorbar(1:height(T), 100*T.accTest, 100*T.accTestStd, 'k.', 'LineWidth',1.2);
        ylabel('测试集分类精度 (%)'); ylim([0 105]); grid on;
    else
        yyaxis left;
        bar(1:height(T), T.mse); ylabel('末次训练 MSE');
        set(gca,'YScale','log');
        yyaxis right;
        errorbar(1:height(T), T.accTest, T.accTestStd, 'o-', 'LineWidth',1.5);
        ylabel('测试集分类精度'); ylim([0 1.02]);
        grid on;
    end
    set(gca,'XTick',1:height(T),'XTickLabel',labels,'XTickLabelRotation',25);
    title(sprintf('(b) %s：结果对比', grpName{g}));
    figs{end+1} = fig_utils(fig, sprintf('exp1_params_%d_%s', g, grpTag(g)));
end

Ta = vertcat(allT{:});
Ta = movevars(Ta, 'group', 'Before', 'label');

R.T   = Ta;
R.fig = figs;
if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    Tzh = Ta;
    Tzh.Properties.VariableNames = {'参数组','取值','MSE','训练精度','测试精度', ...
        '测试精度_标准差','迭代次数','迭代_标准差','达标率','耗时_s','停止原因'};
    writetable(Tzh, fullfile(outdir,'exp1_params.xlsx'), 'WriteMode','replacefile');
end
fprintf('\n参数部分完成。\n');
end

% ------------------------------------------------------------------
function tag = grpTag(g)
tags = {'lr','momentum','activation','init','algorithm','output','stability'};
if g <= numel(tags)
    tag = tags{g};
else
    tag = sprintf('g%d', g);
end
end

%% ===================== 单变量扫描（原 sweep_eval.m） =====================
function [T, curves, labels] = local_sweep(base, variants, D, R)
%SWEEP_EVAL 批量"单变量"对比：对每个变体用 R 个随机种子重复训练并汇总统计
%
%   base     —— 公共设置结构体，必须含 sizes，其余字段见 train_eval
%   variants —— 元胞，每个元素是含 label 与"待覆盖字段"的结构体
%               （可覆盖 sizes/act/outAct/init/wRange/method/lr/mc/maxEpoch/goal/loss）
%   D        —— 数据：Xtr,Ttr,ytr,Xte,yte
%   R        —— 重复次数
%
%   T      —— 汇总表：label,mse,accTrain,accTest,epochs,hitRate,time 及标准差
%   curves —— 第 1 个种子的误差下降曲线
%   labels —— 变体名称

nV     = numel(variants);
rows   = cell(nV,1);
curves = cell(1,nV);
labels = cell(1,nV);

for i = 1:nV
    v = variants{i};
    mse  = zeros(R,1);  aTr = zeros(R,1);  aTe = zeros(R,1);
    ep   = zeros(R,1);  tm  = zeros(R,1);  hit = false(R,1);
    st   = cell(R,1);

    for r = 1:R
        o = base;
        f = fieldnames(v);
        for k = 1:numel(f)
            if ~strcmp(f{k}, 'label')
                o.(f{k}) = v.(f{k});
            end
        end
        o.seed = r;
        m = train_eval(o.sizes, D.Xtr, D.Ttr, D.ytr, D.Xte, D.yte, o);
        mse(r) = m.mse;   aTr(r) = m.accTrain;  aTe(r) = m.accTest;
        ep(r)  = m.epochs; tm(r) = m.time;      hit(r) = strcmp(m.stop,'goal');
        st{r}  = m.stop;
        if r == 1, curves{i} = m.perf(:)'; end
    end

    labels{i} = v.label;
    rows{i} = table({v.label}, mean(mse,'omitnan'), mean(aTr,'omitnan'), ...
        mean(aTe,'omitnan'), std(aTe,'omitnan'), mean(ep,'omitnan'), ...
        std(ep,'omitnan'), mean(hit), mean(tm), {strjoin(unique(st),'/')}, ...
        'VariableNames', {'label','mse','accTrain','accTest','accTestStd', ...
        'epochs','epochsStd','hitRate','time','stop'});
end

T = vertcat(rows{:});
end
