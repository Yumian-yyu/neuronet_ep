function p = fig_utils(fig, name)
%FIG_UTILS 统一的绘图工具（合并原 fig_setup.m 与 save_fig.m）
%
%   fig_utils()              设置全局字体、字号、线宽，保证中文正常显示
%   p = fig_utils(fig,name)  把图窗导出为 figures/name.png（200 dpi）并关闭

if nargin == 0
    p = local_font();
    return
end

outdir = fullfile(project_root(), 'figures');
if ~exist(outdir, 'dir'), mkdir(outdir); end
p = fullfile(outdir, [name '.png']);
exportgraphics(fig, p, 'Resolution', 200);
close(fig);
end

% ------------------------------------------------------------------
function f = local_font()
% 选一个系统里可用的中文字体，避免中文显示成方块
persistent cached
if isempty(cached)
    cand = {'Microsoft YaHei UI', 'Microsoft YaHei', 'Noto Sans SC', ...
            'Source Han Sans SC', 'SimHei', 'SimSun', ...
            'Microsoft JhengHei', 'Arial Unicode MS', 'Arial'};
    avail = listfonts;
    cached = 'Arial';
    for i = 1:numel(cand)
        if any(strcmpi(avail, cand{i}))
            cached = cand{i};
            break;
        end
    end
end

f = cached;
set(groot, 'defaultAxesFontName',   f);
set(groot, 'defaultTextFontName',   f);
set(groot, 'defaultLegendFontName', f);
set(groot, 'defaultAxesFontSize',   10);
set(groot, 'defaultTextFontSize',   10);
set(groot, 'defaultAxesLineWidth',  1.0);
set(groot, 'defaultLineLineWidth',  1.6);
set(groot, 'defaultFigureColor',    'w');
set(groot, 'defaultAxesBox',        'on');
set(groot, 'defaultAxesTickDir',    'out');
end
