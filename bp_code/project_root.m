function r = project_root()
%PROJECT_ROOT 返回项目根目录（即 bp_code 的上一级）
%   所有脚本都通过它定位 figures / results / 实验一参考数据，
%   这样无论文件放在哪个子目录里，路径都不会失效。
r = fileparts(fileparts(mfilename('fullpath')));
end
