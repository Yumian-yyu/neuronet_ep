function R = run_exp2_data(cfg)
%RUN_EXP2_DATA 实验二：不同训练数据对相同拓扑结构 BP 网络建模的影响
%
%   对应实验要求第 2 句：
%     "观察并分析不同训练数据对相同拓扑结构 BP 网络建模的影响"
%
%   固定 4-8-3 结构，只改变训练数据：
%     样本数量 / 归一化方式 / 训练数据噪声 / 类别均衡性 / 样本输入顺序 / 特征子集

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 5;   end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
fprintf('\n==================================================================\n');
fprintf('  实验二：不同训练数据对相同拓扑结构（4-8-3）建模的影响\n');
fprintf('==================================================================\n');

R = local_data(struct('R', cfg.R, 'save', cfg.save));

% ---------- 小结 ----------
fprintf('\n【实验二 小结】\n');
fprintf('  样本量：每类 2 个 → 每类 25 个，测试精度 %.1f%% → %.1f%%\n', ...
        100*R.rec.nAccTe(1), 100*R.rec.nAccTe(end));
fprintf('  归一化：不归一化 %.1f%%，z-score %.1f%%\n', ...
        100*R.rec.nmAcc(1), 100*R.rec.nmAcc(3));
fprintf('  噪声：sigma=0 → 1.5，精度 %.1f%% → %.1f%%\n', ...
        100*R.rec.nzAcc(1), 100*R.rec.nzAcc(end));
fprintf('  特征：全 4 维 %.1f%%，仅花瓣 %.1f%%，仅花萼 %.1f%%\n', ...
        100*R.rec.fAcc(1), 100*R.rec.fAcc(2), 100*R.rec.fAcc(3));
end

%% ===================== 训练数据扫描（原 data_sweep.m） =====================
function R = local_data(cfg)
%DATA_SWEEP 实验一 内容(2)：不同训练数据对相同拓扑结构 BP 网络建模的影响
%
%   固定网络结构 4-8-3（与训练算法、迭代上限、初始化方式都不变），
%   只改变"喂给网络的训练数据"，考察六种因素：
%     (1) 训练样本数量      每类 5/8/10/15/20/25 个
%     (2) 输入归一化方式    不归一化 / min-max / z-score
%     (3) 训练数据噪声      高斯噪声 sigma = 0/0.05/0.1/0.15/0.2（以标准差为单位）
%     (4) 类别不均衡        均衡 / 少类 / 缺类
%     (5) 样本输入顺序      按类别排序 / 随机打乱（增量学习）
%     (6) 特征子集          全部4维 / 只用花瓣 / 只用花萼
%
%   每个设置用 5 个随机种子重复，报告均值±标准差以及各类召回率。

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'),    cfg.R = 5;     end
if ~isfield(cfg,'save'), cfg.save = true; end

fig_utils();
S  = load_iris();
Rr = cfg.R;

o0 = struct('sizes',[4 8 3], 'method','gda', 'lr',0.1, 'mc',0.9, ...
            'act','logsig', 'outAct','logsig', 'init','nguyen', ...
            'maxEpoch',2000, 'goal',1e-3, 'loss','mse');

rows = cell(0,1);
rec  = struct();   % 收集画图用的数据

%% ================= (1) 训练样本数量 =================
fprintf('\n===== (1) 训练样本数量 =====\n');
P  = norm_fit(S.train.X, 'zscore');
XtrAll = norm_apply(S.train.X, P);   Xte = norm_apply(S.test.X, P);
nList = [2 3 5 8 12 18 25];
nAccTe = zeros(numel(nList),1); nAccTeS = nAccTe; nAccTr = nAccTe;
for k = 1:numel(nList)
    n = nList(k);
    aTe = zeros(Rr,1); aTr = zeros(Rr,1); ep = zeros(Rr,1);
    for r = 1:Rr
        rng(100+r);                        % 每个种子抽取不同的训练子集
        idx = [];
        for c = 1:3
            cand = find(S.train.y == c);
            idx = [idx, cand(randperm(numel(cand), n))];   %#ok<AGROW>
        end
        idx = idx(randperm(numel(idx)));
        o = o0;  o.seed = r;
        m = train_eval(o.sizes, XtrAll(:,idx), S.train.T(:,idx), S.train.y(idx), Xte, S.test.y, o);
        aTe(r) = m.accTest;  aTr(r) = m.accTrain;  ep(r) = m.mse;
    end
    nAccTe(k) = mean(aTe);  nAccTeS(k) = std(aTe);  nAccTr(k) = mean(aTr);
    rows{end+1} = mkrow('训练样本量', sprintf('每类%d个', n), ...
        mean(aTe), std(aTe), mean(aTr), mean(ep), NaN(1,3), NaN);  %#ok<SAGROW>
    fprintf('  每类%2d个(共%3d)：测试精度=%.4f  训练精度=%.4f  末MSE=%.2e\n', ...
            n, 3*n, mean(aTe), mean(aTr), mean(ep));
end
rec.nList = nList; rec.nAccTe = nAccTe; rec.nAccTeS = nAccTeS; rec.nAccTr = nAccTr;

%% ================= (2) 归一化方式 =================
fprintf('\n===== (2) 输入归一化方式 =====\n');
norms = {'none','minmax','zscore'};
zname = {'不归一化','min-max','z-score'};
nmAcc = zeros(1,3); nmAccS = nmAcc; nmMse = nmAcc; nmEp = nmAcc;
for k = 1:numel(norms)
    Pk  = norm_fit(S.train.X, norms{k});
    Xa  = norm_apply(S.train.X, Pk);
    Xb  = norm_apply(S.test.X,  Pk);
    aTe = zeros(Rr,1); mse = zeros(Rr,1); ep = zeros(Rr,1);
    for r = 1:Rr
        o = o0;  o.seed = r;
        m = train_eval(o.sizes, Xa, S.train.T, S.train.y, Xb, S.test.y, o);
        aTe(r) = m.accTest;  mse(r) = m.mse;  ep(r) = m.epochs;
    end
    nmAcc(k) = mean(aTe);  nmAccS(k) = std(aTe);  nmMse(k) = mean(mse);  nmEp(k) = mean(ep);
    rows{end+1} = mkrow('归一化方式', zname{k}, mean(aTe), std(aTe), NaN, mean(mse), NaN(1,3), mean(ep)); %#ok<SAGROW>
    fprintf('  %-10s 测试精度=%.4f  末MSE=%.2e  迭代=%4.0f\n', ...
            zname{k}, mean(aTe), mean(mse), mean(ep));
end
rec.nmAcc = nmAcc; rec.nmAccS = nmAccS; rec.zname = zname;

%% ================= (3) 训练数据噪声 =================
fprintf('\n===== (3) 训练数据噪声 =====\n');
sigList = [0 0.2 0.4 0.6 0.8 1.0 1.5];
nzAcc = zeros(numel(sigList),1); nzAccS = nzAcc; nzAccTr = nzAcc;
P  = norm_fit(S.train.X, 'zscore');
Xtr0 = norm_apply(S.train.X, P);  Xte = norm_apply(S.test.X, P);
for k = 1:numel(sigList)
    sg = sigList(k);
    aTe = zeros(Rr,1); aTr = zeros(Rr,1);
    for r = 1:Rr
        rng(300+r);                        % 每个种子加不同噪声
        Xn = Xtr0 + sg * randn(size(Xtr0));
        o = o0;  o.seed = r;
        m = train_eval(o.sizes, Xn, S.train.T, S.train.y, Xte, S.test.y, o);
        aTe(r) = m.accTest;  aTr(r) = m.accTrain;
    end
    nzAcc(k) = mean(aTe);  nzAccS(k) = std(aTe);  nzAccTr(k) = mean(aTr);
    rows{end+1} = mkrow('训练数据噪声', sprintf('sigma=%.2f', sg), ...
        mean(aTe), std(aTe), mean(aTr), NaN, NaN(1,3), NaN);  %#ok<SAGROW>
    fprintf('  sigma=%.2f：测试精度=%.4f  训练精度=%.4f\n', sg, mean(aTe), mean(aTr));
end
rec.sigList = sigList; rec.nzAcc = nzAcc; rec.nzAccS = nzAccS; rec.nzAccTr = nzAccTr;

%% ================= (4) 类别不均衡 =================
fprintf('\n===== (4) 类别不均衡 =====\n');
imb = {[25 25 25], [25 15 5], [25 5 2]};
iname = {'均衡 25/25/25','不均衡 25/15/5','不均衡 25/5/2'};
imAcc = zeros(1,3); imRec = zeros(3,3); imAccS = imAcc;
for k = 1:numel(imb)
    cnt = imb{k};
    idx = [];
    for c = 1:3
        cand = find(S.train.y == c);
        idx = [idx, cand(1:cnt(c))];   %#ok<AGROW>
    end
    rl = zeros(3,Rr); aTe = zeros(Rr,1);
    for r = 1:Rr
        o = o0;  o.seed = r;
        m = train_eval(o.sizes, Xtr0(:,idx), S.train.T(:,idx), S.train.y(idx), ...
                       Xte, S.test.y, o);
        yh = bp_predict(m.net, Xte);
        aTe(r) = mean(yh == S.test.y);
        for c = 1:3
            sel = (S.test.y == c);
            rl(c,r) = mean(yh(sel) == c);
        end
    end
    imAcc(k) = mean(aTe);  imAccS(k) = std(aTe);  imRec(:,k) = mean(rl,2);
    rows{end+1} = mkrow('类别不均衡', iname{k}, mean(aTe), std(aTe), NaN, NaN, ...
        mean(rl,2)', NaN);   %#ok<SAGROW>
    fprintf('  %-16s 总体精度=%.4f  各类召回=%.3f/%.3f/%.3f\n', ...
            iname{k}, mean(aTe), mean(rl(1,:)), mean(rl(2,:)), mean(rl(3,:)));
end
rec.imAcc = imAcc; rec.imAccS = imAccS; rec.imRec = imRec; rec.iname = iname;

%% ================= (5) 样本输入顺序（增量学习） =================
fprintf('\n===== (5) 样本输入顺序（增量学习, 30 轮）=====\n');
oI = o0;  oI.method = 'inc';  oI.maxEpoch = 30;
[~, ord] = sort(S.train.y);          % 按类别排序
shuf = randperm(numel(S.train.y));
seAcc = zeros(2,Rr); seCurve = cell(2,1);
for k = 1:2
    if k == 1, idx = ord;  nm = '按类别排序';
    else,      idx = shuf; nm = '随机打乱'; end
    for r = 1:Rr
        o = oI;  o.seed = r;
        m = train_eval(o.sizes, Xtr0(:,idx), S.train.T(:,idx), S.train.y(idx), ...
                       Xte, S.test.y, o);
        seAcc(k,r) = m.accTest;
        if r == 1, seCurve{k} = m.perf; end
    end
    rows{end+1} = mkrow('样本输入顺序', nm, mean(seAcc(k,:)), std(seAcc(k,:)), ...
        NaN, NaN, NaN(1,3), NaN);   %#ok<SAGROW>
    fprintf('  %-12s 测试精度=%.4f\n', nm, mean(seAcc(k,:)));
end
rec.seAcc = seAcc; rec.seCurve = seCurve;

%% ================= (6) 特征子集 =================
fprintf('\n===== (6) 输入特征子集 =====\n');
feats = {1:4, 3:4, 1:2};
fname = {'全部4维','仅花瓣(第3,4维)','仅花萼(第1,2维)'};
fAcc = zeros(1,3); fAccS = fAcc;
for k = 1:numel(feats)
    f = feats{k};
    o = o0;  o.sizes = [numel(f) 8 3];
    aTe = zeros(Rr,1);
    for r = 1:Rr
        oo = o;  oo.seed = r;
        m = train_eval(oo.sizes, Xtr0(f,:), S.train.T, S.train.y, Xte(f,:), S.test.y, oo);
        aTe(r) = m.accTest;
    end
    fAcc(k) = mean(aTe);  fAccS(k) = std(aTe);
    rows{end+1} = mkrow('特征子集', fname{k}, mean(aTe), std(aTe), NaN, NaN, NaN(1,3), NaN); %#ok<SAGROW>
    fprintf('  %-16s 测试精度=%.4f\n', fname{k}, mean(aTe));
end
rec.fAcc = fAcc; rec.fAccS = fAccS; rec.fname = fname;

%% ================= 汇总与作图 =================
T = vertcat(rows{:});
R.T = T;

% ---- 图 1：样本量 + 归一化 ----
fig = figure('Visible','off','Position',[80 80 1060 400]);
subplot(1,2,1);
errorbar(3*nList, 100*nAccTe, 100*nAccTeS, '-o','MarkerFaceColor','b','CapSize',4); hold on;
plot(3*nList, 100*nAccTr, '--s', 'MarkerFaceColor',[0.85 0.33 0.10],'MarkerSize',6);
grid on; xlabel('训练样本总数'); ylabel('分类精度 (%)');
title('(a) 训练样本数量'); legend({'测试集','训练集'},'Location','southeast');
subplot(1,2,2);
bb = bar(100*nmAcc); bb.FaceColor = [0.30 0.55 0.85]; hold on;
errorbar(1:3, 100*nmAcc, 100*nmAccS, 'k.', 'LineWidth',1.2);
set(gca,'XTick',1:3,'XTickLabel',zname);
ylabel('测试集分类精度 (%)'); ylim([0 100]); grid on;
title('(b) 输入归一化方式');
R.fig1 = fig_utils(fig, 'exp2_data_size_norm');

% ---- 图 2：噪声 + 特征子集 ----
fig = figure('Visible','off','Position',[80 80 1060 400]);
subplot(1,2,1);
errorbar(sigList, 100*nzAcc, 100*nzAccS, '-o','MarkerFaceColor','b','CapSize',4); hold on;
plot(sigList, 100*nzAccTr, '--s','MarkerFaceColor',[0.85 0.33 0.10],'MarkerSize',6);
grid on; xlabel('训练数据高斯噪声标准差 \sigma'); ylabel('分类精度 (%)');
title('(a) 训练数据噪声'); legend({'测试集','训练集'},'Location','northeast');
subplot(1,2,2);
bb = bar(100*fAcc); bb.FaceColor = [0.30 0.55 0.85]; hold on;
errorbar(1:3, 100*fAcc, 100*fAccS, 'k.', 'LineWidth',1.2);
set(gca,'XTick',1:3,'XTickLabel',fname,'XTickLabelRotation',15);
ylabel('测试集分类精度 (%)'); ylim([0 100]); grid on;
title('(b) 输入特征子集');
R.fig2 = fig_utils(fig, 'exp2_data_noise_feature');

% ---- 图 3：类别不均衡 + 样本顺序 ----
fig = figure('Visible','off','Position',[80 80 1060 420]);
subplot(1,2,1);
bar(100*imRec.'); hold on;
set(gca,'XTick',1:3,'XTickLabel',iname,'XTickLabelRotation',15);
ylabel('各类召回率 (%)'); ylim([0 105]); grid on;
legend({'类别1','类别2','类别3'},'Location','southwest');
title('(a) 类别不均衡的影响');
subplot(1,2,2);
hold on;
plot(seCurve{1}, '-', 'LineWidth',1.6);
plot(seCurve{2}, '--', 'LineWidth',1.6);
grid on; ylim([0.08 0.40]);
xlabel('迭代次数'); ylabel('训练均方误差 MSE');
title('(b) 样本输入顺序（增量学习）');
legend({'按类别排序','随机打乱'},'Location','northeast');
R.fig3 = fig_utils(fig, 'exp2_data_imbalance_order');

R.rec = rec;
if cfg.save
    outdir = fullfile(project_root(), 'results');
    if ~exist(outdir,'dir'), mkdir(outdir); end
    Tzh = T;
    Tzh.Properties.VariableNames = {'影响因素','设置','测试精度','测试精度_标准差', ...
        '训练精度','末次MSE','召回率_类1','召回率_类2','召回率_类3','达标迭代'};
    writetable(Tzh, fullfile(outdir,'exp2_data.xlsx'), 'WriteMode','replacefile');
end
fprintf('\n训练数据部分完成。\n');
end

% ------------------------------------------------------------------
function t = mkrow(fac, set, accTe, accTeS, accTr, mse, recall, ep)
t = table({fac}, {set}, accTe, accTeS, accTr, mse, ...
          recall(1), recall(2), recall(3), ep, ...
          'VariableNames', {'factor','setting','accTest','accTestStd','accTrain', ...
          'mse','rec1','rec2','rec3','epochs'});
end
