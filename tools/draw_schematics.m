function draw_schematics()
%DRAW_SCHEMATICS 生成实验报告用的示意图
%   图1：4-8-3 BP 网络拓扑结构
%   图2：BP 算法训练流程图
%   输出到 figures/fig_bp_structure.png 与 figures/fig_flowchart.png

local_fonts();

%% ================= 图 1：网络结构 =================
fig = figure('Visible','off','Position',[80 80 940 620]);
axes('Position',[0 0 1 1]); axis off; hold on;
axis([0 1.12 0 1]);

nIn = 4; nHid = 8; nOut = 3;
xIn = 0.15; xHid = 0.50; xOut = 0.82;
yIn  = linspace(0.74, 0.20, nIn);
yHid = linspace(0.86, 0.10, nHid);
yOut = linspace(0.66, 0.28, nOut);

for i = 1:nIn
    for j = 1:nHid
        plot([xIn xHid], [yIn(i) yHid(j)], '-', 'Color',[0.76 0.76 0.76], 'LineWidth',0.6);
    end
end
for j = 1:nHid
    for k = 1:nOut
        plot([xHid xOut], [yHid(j) yOut(k)], '-', 'Color',[0.76 0.76 0.76], 'LineWidth',0.6);
    end
end

draw_nodes(xIn,  yIn,  nIn,  [0.24 0.60 0.88], 0.028);
draw_nodes(xHid, yHid, nHid, [0.45 0.80 0.45], 0.026);
draw_nodes(xOut, yOut, nOut, [0.95 0.62 0.42], 0.028);

inNames  = {'花萼长','花萼宽','花瓣长','花瓣宽'};
outNames = {'类1 山鸢尾','类2 变色鸢尾','类3 维吉尼亚'};
for i = 1:nIn
    text(xIn-0.045, yIn(i), inNames{i}, 'HorizontalAlignment','right', ...
         'VerticalAlignment','middle', 'FontSize',11);
end
for k = 1:nOut
    text(xOut+0.045, yOut(k), outNames{k}, 'HorizontalAlignment','left', ...
         'VerticalAlignment','middle', 'FontSize',11);
end

text(xIn,  0.92, '输入层', 'HorizontalAlignment','center','FontSize',13,'FontWeight','bold');
text(xHid, 0.96, '隐层（8 个神经元，logsig）', 'HorizontalAlignment','center', ...
     'FontSize',13,'FontWeight','bold');
text(xOut, 0.92, '输出层', 'HorizontalAlignment','center','FontSize',13,'FontWeight','bold');

text((xIn+xHid)/2, 0.035, 'W^{(1)}: 8×4， b^{(1)}: 8×1', ...
     'HorizontalAlignment','center','FontSize',11);
text((xHid+xOut)/2, 0.035, 'W^{(2)}: 3×8， b^{(2)}: 3×1', ...
     'HorizontalAlignment','center','FontSize',11);

title('BP 神经网络拓扑结构（4-8-3）', 'FontSize',15);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), '..', 'figures', ...
    'fig_bp_structure.png'), 'Resolution', 200);
close(fig);

%% ================= 图 2：训练流程图 =================
fig = figure('Visible','off','Position',[80 80 760 1000]);
axes('Position',[0 0 1 1]); axis off; hold on;
axis([0 1 0 1]);

cx = 0.42;  bw = 0.60;  bh = 0.062;

steps = {
 '开  始'
 '读入训练/测试数据，z-score 归一化，标签 one-hot 编码'
 '按 Nguyen-Widrow 规则随机初始化权值 W、b'
 '前向传播：a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})'
 '计算训练误差：E = (1/N)·Σ ½‖t − y‖²'
 '误差反传：δ^{(L)} = (y−t)⊙f''(z)'
 '　　　　　δ^{(l)} = (W^{(l+1)})^{T}δ^{(l+1)}⊙f''(z^{(l)})'
 '按所选算法更新权值：W ← W − η·δ·a^{T}（可带动量/自适应 η）'
 '迭代次数 + 1'
};

yStep = [0.955 0.870 0.780 0.685 0.595 0.505 0.425 0.340 0.262];

for i = 1:numel(steps)
    draw_box(cx, yStep(i), bw, bh, steps{i}, i==1);
    if i > 1
        draw_arrow(cx, yStep(i-1)-bh/2, cx, yStep(i)+bh/2);
    end
end

% 判定菱形
yDec = 0.165;
dh   = 0.062;  dw = 0.155;
patch([cx cx+dw cx cx-dw], [yDec+dh yDec yDec-dh yDec], [1 0.97 0.80], ...
      'EdgeColor','k', 'LineWidth',1);
text(cx, yDec, 'E < 目标误差 ?', 'HorizontalAlignment','center', ...
     'VerticalAlignment','middle', 'FontSize',11);
draw_arrow(cx, yStep(end)-bh/2, cx, yDec+dh);

% “否”：右侧回路回到前向传播
xr = 0.94;
plot([cx+dw xr], [yDec yDec], 'k-', 'LineWidth',1);
plot([xr xr], [yDec yStep(4)], 'k-', 'LineWidth',1);
draw_arrow(xr, yStep(4), cx+bw/2, yStep(4));
text((cx+dw+xr)/2, yDec+0.022, '否', 'FontSize',11, 'HorizontalAlignment','center');

% “是”：向下输出结果
yOutBox = 0.045;
draw_arrow(cx, yDec-dh, cx, yOutBox+bh/2);
draw_box(cx, yOutBox, bw*0.72, bh, '输出：分类精度、混淆矩阵、误差曲线', true);
text(cx+0.03, (yDec-dh+yOutBox+bh/2)/2, '是', 'FontSize',11);

title('BP 算法训练流程', 'FontSize',15);
exportgraphics(fig, fullfile(fileparts(mfilename('fullpath')), '..', 'figures', ...
    'fig_flowchart.png'), 'Resolution', 200);
close(fig);

fprintf('示意图已生成。\n');
end

% ------------------------------------------------------------------
function draw_nodes(x, yv, n, col, r)
for i = 1:n
    rectangle('Position',[x-r, yv(i)-r, 2*r, 2*r], 'Curvature',[1 1], ...
              'FaceColor',col, 'EdgeColor',[0.20 0.20 0.20], 'LineWidth',1);
end
end

function draw_box(cx, cy, w, h, str, bold)
if bold
    fc = [0.90 0.93 0.98];  fw = 'bold';
else
    fc = [1 1 1];           fw = 'normal';
end
rectangle('Position',[cx-w/2, cy-h/2, w, h], 'Curvature',[0.10 0.30], ...
          'FaceColor',fc, 'EdgeColor','k', 'LineWidth',1);
text(cx, cy, str, 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontSize',10, 'FontWeight',fw);
end

function draw_arrow(x1, y1, x2, y2)
annotation('arrow', [x1 x2], [y1 y2]);
end

% ------------------------------------------------------------------
function local_fonts()
% 与 bp_code/tools/fig_utils.m 一致的中文字体设置，保证本脚本可独立运行
persistent cached
if isempty(cached)
    cand = {'Microsoft YaHei UI','Microsoft YaHei','Noto Sans SC', ...
            'Source Han Sans SC','SimHei','SimSun','Microsoft JhengHei','Arial'};
    avail = listfonts;
    cached = 'Arial';
    for i = 1:numel(cand)
        if any(strcmpi(avail, cand{i})), cached = cand{i}; break; end
    end
end
set(groot, 'defaultAxesFontName',   cached);
set(groot, 'defaultTextFontName',   cached);
set(groot, 'defaultLegendFontName', cached);
set(groot, 'defaultAxesFontSize',   10);
set(groot, 'defaultTextFontSize',   10);
set(groot, 'defaultAxesLineWidth',  1.0);
set(groot, 'defaultLineLineWidth',  1.6);
set(groot, 'defaultFigureColor',    'w');
set(groot, 'defaultAxesBox',        'on');
set(groot, 'defaultAxesTickDir',    'out');
end
