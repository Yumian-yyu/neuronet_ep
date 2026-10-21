function main(cfg)
%MAIN 一键运行神经网络实验的全部三项内容
%
%   main                  % 每个配置重复 5 个随机种子（约 80 秒）
%   main(struct('R',3))   % 快速版，重复次数少，标准差偏大
%
%   三项实验严格对应实验要求的三句话：
%     实验一  拓扑结构与各项参数对训练结果的影响
%     实验二  不同训练数据对相同拓扑结构建模的影响
%     实验三  设计实现 BP 网络完成识别分类，并按课堂方法检验
%
%   输出：../figures/ 与 ../results/

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'R'), cfg.R = 5; end

here = fileparts(mfilename('fullpath'));
addpath(genpath(here));
fig_utils();

fprintf('==================================================\n');
fprintf('  神经网络实验：BP 网络建模与 Iris 识别分类\n');
fprintf('  每个配置重复 %d 个随机种子，MATLAB %s\n', cfg.R, version);
fprintf('==================================================\n');

tAll = tic;
logLines = {sprintf('运行时间 %s', datestr(now,'yyyy-mm-dd HH:MM:SS'))};

t = tic;
R1 = run_exp1_network(struct('R', cfg.R, 'save', true));
logLines{end+1} = sprintf('实验一（拓扑+参数）：%.1f s', toc(t));

t = tic;
R2 = run_exp2_data(struct('R', cfg.R, 'save', true));
logLines{end+1} = sprintf('实验二（训练数据）：%.1f s', toc(t));

t = tic;
R3 = run_exp3_classify(struct('R', 10, 'K', 5, 'save', true));
logLines{end+1} = sprintf('实验三（分类+检验）：%.1f s', toc(t));

% ---------- 汇总 ----------
fprintf('\n==================================================\n');
fprintf('全部实验完成，总耗时 %.1f s\n', toc(tAll));

Tz = R1.topology.T(R1.topology.T.norm == "zscore", :);
[~, iBest] = max(Tz.accTest);
fprintf('实验一 最优结构：%s（测试精度 %.2f%%）\n', Tz.topo{iBest}, 100*Tz.accTest(iBest));
fprintf('实验二 特征子集：全 4 维 %.1f%%，仅花瓣 %.1f%%，仅花萼 %.1f%%\n', ...
        100*R2.rec.fAcc(1), 100*R2.rec.fAcc(2), 100*R2.rec.fAcc(3));
fprintf('实验三 分类精度：%.2f%% ± %.2f%%（%d 次重复）\n', ...
        100*R3.classify.accMean, 100*R3.classify.accStd, numel(R3.classify.acc));
fprintf('实验三 宏平均：精确率 %.2f%%  召回率 %.2f%%  F1 %.2f%%\n', ...
        100*mean(R3.classify.metrics.precision), ...
        100*mean(R3.classify.metrics.recall), ...
        100*mean(R3.classify.metrics.f1));
fprintf('实验三 5 折交叉验证：%.2f%% ± %.2f%%\n', ...
        100*mean(R3.cv.acc), 100*std(R3.cv.acc));
fprintf('实验三 宏平均 AUC：%.4f\n', R3.roc.m.macroAUC);
fprintf('结果表：%s\n', fullfile(here,'..','results'));
fprintf('图片：  %s\n', fullfile(here,'..','figures'));
fprintf('==================================================\n');

outdir = fullfile(project_root(), 'results');
if ~exist(outdir,'dir'), mkdir(outdir); end
fid = fopen(fullfile(outdir,'run_log.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s\n', logLines{:});
fprintf(fid, '实验一 最优结构：%s（测试精度 %.2f%%）\n', Tz.topo{iBest}, 100*Tz.accTest(iBest));
fprintf(fid, '实验三 分类精度：%.2f%% ± %.2f%%\n', 100*R3.classify.accMean, 100*R3.classify.accStd);
fprintf(fid, '实验三 5 折交叉验证：%.2f%% ± %.2f%%\n', 100*mean(R3.cv.acc), 100*std(R3.cv.acc));
fprintf(fid, '实验三 宏平均 AUC：%.4f\n', R3.roc.m.macroAUC);
fclose(fid);
end
