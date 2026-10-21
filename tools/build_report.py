# -*- coding: utf-8 -*-
"""根据 5.实验报告模板.docx 生成神经网络实验报告。"""
import os
import openpyxl
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = r"D:\matlab_projects\神经网路实验"
TPL = os.path.join(ROOT, "5. 实验报告模板.docx")
OUT = os.path.join(ROOT, "神经网络实验报告.docx")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
ASCII_FONT = "Times New Roman"


# ---------------------------------------------------------------- 工具函数
def set_run(run, size=12, bold=False, cn=BODY_FONT, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = ASCII_FONT
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), ASCII_FONT)
    rFonts.set(qn("w:hAnsi"), ASCII_FONT)
    rFonts.set(qn("w:eastAsia"), cn)
    if color:
        run.font.color.rgb = color
    return run


def para_format(p, first_indent=0.0, before=0, after=0, line=1.4, align=None):
    pf = p.paragraph_format
    pf.first_line_indent = Cm(first_indent)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        pf.alignment = align
    # 模板启用了中文文档网格，段落若对齐到网格，行距会被放大 2~3 倍
    pPr = p._p.get_or_add_pPr()
    if pPr.find(qn("w:snapToGrid")) is None:
        snap = OxmlElement("w:snapToGrid")
        snap.set(qn("w:val"), "0")
        pPr.insert_element_before(
            snap, "w:spacing", "w:ind", "w:jc", "w:textDirection",
            "w:textAlignment", "w:outlineLvl", "w:rPr", "w:sectPr")
    return p


class CellWriter:
    """在大单元格中按顺序插入段落、表格和图片。"""

    def __init__(self, cell):
        self.cell = cell
        self.cursor = None
        self.fig_cell = None

    def _place(self, element):
        if hasattr(element, "_p"):
            el = element._p
        elif hasattr(element, "_tbl"):
            el = element._tbl
        else:
            el = element
        if self.cursor is None:
            first = self.cell.paragraphs[0]
            first._p.addnext(el)
            self.cursor = first
        else:
            if hasattr(self.cursor, "_tbl"):
                self.cursor._tbl.addnext(el)
            else:
                self.cursor._p.addnext(el)

    def h2(self, text):
        p = self.cell.add_paragraph()
        set_run(p.add_run(text), size=12, bold=True, cn=HEAD_FONT)
        para_format(p, first_indent=0.0, before=8, after=4, line=1.35)
        self._place(p)
        self.cursor = p
        return p

    def body(self, text, indent=0.74, size=12, bold=False, before=0, after=2,
             align=None, line=1.45):
        p = self.cell.add_paragraph()
        set_run(p.add_run(text), size=size, bold=bold)
        para_format(p, first_indent=indent, before=before, after=after,
                    line=line, align=align)
        self._place(p)
        self.cursor = p
        return p

    def formula(self, text):
        p = self.cell.add_paragraph()
        set_run(p.add_run(text), size=11.5)
        para_format(p, first_indent=0.0, before=2, after=2, line=1.3,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
        self._place(p)
        self.cursor = p
        return p

    def caption(self, text):
        # 图片被放进 1x1 无边框表格（行不可拆分），图注写在同一单元格内，
        # 从而在任何渲染器里都不会与图分到两页。
        if self.fig_cell is not None:
            tc = self.fig_cell
            p = tc.add_paragraph()
            set_run(p.add_run(text), size=10.5, cn=HEAD_FONT)
            para_format(p, first_indent=0.0, before=3, after=6, line=1.2,
                        align=WD_ALIGN_PARAGRAPH.CENTER)
            self.fig_cell = None
            return p

        p = self.cell.add_paragraph()
        set_run(p.add_run(text), size=10.5, cn=HEAD_FONT)
        para_format(p, first_indent=0.0, before=6, after=2, line=1.2,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
        self._place(p)
        self.cursor = p
        return p

    def table_caption(self, text):
        p = self.cell.add_paragraph()
        set_run(p.add_run(text), size=10.5, cn=HEAD_FONT)
        para_format(p, first_indent=0.0, before=8, after=3, line=1.2,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
        p.paragraph_format.keep_with_next = True
        self._place(p)
        self.cursor = p
        return p

    def picture(self, filename, width_cm=14.6):
        """把图片放进一个 1x1 无边框表格中，配合 cantSplit 保证图与图注不被分页拆开。"""
        path = os.path.join(FIG, filename)
        tbl = self.cell.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        tc = tbl.rows[0].cells[0]
        tc.width = Cm(15.0)
        set_cell_margins(tc, 0)
        trPr = tbl.rows[0]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))

        p = tc.paragraphs[0]
        p.add_run().add_picture(path, width=Cm(width_cm))
        para_format(p, first_indent=0.0, before=2, after=0, line=1.0,
                    align=WD_ALIGN_PARAGRAPH.CENTER)

        self._place(tbl)
        self.cursor = tbl
        self.fig_cell = tc
        return tbl

    def table(self, header, rows, widths, font_size=9, header_size=9,
              align_first_left=True):
        n_col = len(header)
        tbl = self.cell.add_table(rows=1 + len(rows), cols=n_col)
        set_borders(tbl)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        def fill(cell_obj, text, bold=False, size=font_size, center=True):
            cell_obj.text = ""
            p = cell_obj.paragraphs[0]
            set_run(p.add_run(str(text)), size=size, bold=bold)
            para_format(p, first_indent=0.0, before=1, after=1, line=1.0,
                        align=WD_ALIGN_PARAGRAPH.CENTER if center
                        else WD_ALIGN_PARAGRAPH.LEFT)
            tcPr = cell_obj._tc.get_or_add_tcPr()
            tcPr.set(qn("w:vAlign"), "center")

        for j, h in enumerate(header):
            c = tbl.rows[0].cells[j]
            fill(c, h, bold=True, size=header_size)
            shade(c, "DCE6F1")
        for i, row in enumerate(rows, start=1):
            for j, v in enumerate(row):
                fill(tbl.rows[i].cells[j], v,
                     center=not (align_first_left and j == 0))

        # 跨页时重复表头，且禁止单行被拆到两页
        for i, row in enumerate(tbl.rows):
            trPr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trPr.append(cant)
            if i == 0:
                th = OxmlElement("w:tblHeader")
                trPr.append(th)

        for row in tbl.rows:
            for j, c in enumerate(row.cells):
                c.width = Cm(widths[j])

        # 表格后补一个空段落，避免与下一段粘连
        self._place(tbl)
        self.cursor = tbl
        spacer = self.cell.add_paragraph()
        set_run(spacer.add_run(""), size=6)
        para_format(spacer, before=0, after=0, line=1.0)
        self.cursor._tbl.addnext(spacer._p)
        self.cursor = spacer
        return tbl

    def page_break(self):
        p = self.cell.add_paragraph()
        run = p.add_run()
        br = OxmlElement("w:br")
        br.set(qn("w:type"), "page")
        run._element.append(br)
        para_format(p, before=0, after=0, line=1.0)
        self._place(p)
        self.cursor = p


def set_cell_margins(cell_obj, twips=0):
    tcPr = cell_obj._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:w"), str(twips))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def set_borders(tbl):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "404040")
        borders.append(el)
    tblPr.append(borders)


def shade(cell_obj, color):
    tcPr = cell_obj._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


# ---------------------------------------------------------------- 读取数据
def read_xlsx(name):
    wb = openpyxl.load_workbook(os.path.join(RES, name), data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    return rows[0], rows[1:]


def fmt_pct(x, nd=1):
    if x is None:
        return "—"
    return f"{100*float(x):.{nd}f}"


def fmt_e(x, nd=2):
    if x is None:
        return "—"
    x = float(x)
    if x == float("inf"):
        return "Inf"
    if x != x:
        return "—"
    return f"{x:.{nd}e}"


def fmt_i(x):
    if x is None or x != x:
        return "—"
    return f"{float(x):.0f}"


# ---------------------------------------------------------------- 主流程
def main():
    doc = Document(TPL)

    # 封面表格：保留原样，仅提示填写
    cell = doc.tables[1].rows[0].cells[0]

    # 删除单元格中的空段落
    for p in list(cell.paragraphs):
        if not p.text.strip():
            p._p.getparent().remove(p._p)

    # 行高改为“最小值“，避免跨页被截断
    tr = doc.tables[1].rows[0]._tr
    trPr = tr.get_or_add_trPr()
    h = trPr.find(qn("w:trHeight"))
    if h is not None:
        h.set(qn("w:hRule"), "atLeast")

    # 定位各级标题
    heads = {}
    for p in cell.paragraphs:
        t = p.text.strip()
        if t.startswith("实验名称"):
            heads["实验名称"] = p
        elif t.startswith("一、"):
            heads["一、"] = p
        elif t.startswith("二、"):
            heads["二、"] = p
        elif "实验线路示图" in t:
            heads["三、"] = p
        elif t.startswith("四、"):
            heads["四、"] = p
        elif t.startswith("五、"):
            heads["五、"] = p

    # 实验名称：在原标题后补全名称
    p0 = heads["实验名称"]
    runs = p0.runs
    if len(runs) >= 2:
        runs[1].text = "：BP 神经网络建模与 Iris 鸢尾花识别分类"
    else:
        set_run(p0.add_run("：BP 神经网络建模与 Iris 鸢尾花识别分类"), size=16)
    for r in p0.runs:
        set_run(r, size=16, bold=True, cn=HEAD_FONT)

    # 第三个标题补上“三、”
    p3 = heads["三、"]
    if p3.runs:
        p3.runs[0].text = "三、实验线路示图、程序算法流程图"
        for r in p3.runs[1:]:
            r.text = ""

    build_section1(CellWriter(cell), heads["一、"])
    build_section2(CellWriter(cell), heads["二、"])
    build_section3(CellWriter(cell), heads["三、"])
    build_section4(CellWriter(cell), heads["四、"])
    build_section5(CellWriter(cell), heads["五、"])

    doc.save(OUT)
    print("saved:", OUT)


# ---------------------------------------------------------------- 各章节
def build_section1(w, head):
    w.cursor = head

    w.h2("1. 实验内容")
    w.body("（1）设计并实现 BP 神经网络。不直接调用工具箱的训练函数，而是依据 BP 算法公式，"
           "自行编写数据归一化、权值初始化、前向传播、误差反传和权值更新等环节，"
           "形成一套可训练、可预测、可更换训练算法的 BP 网络程序。")
    w.body("（2）观察比较 BP 网络拓扑结构及其它各项参数变化对训练结果的影响。采用控制变量法，"
           "固定数据与其余训练条件，分别改变隐层数、隐层神经元个数、学习率、动量因子、"
           "隐层激活函数、权值初始化方式和训练算法，记录收敛速度、训练误差与测试精度。")
    w.body("（3）观察并分析不同训练数据对相同拓扑结构 BP 网络建模的影响。固定 4-8-3 拓扑结构，"
           "改变训练样本数量、输入归一化方式、训练数据噪声、类别分布、样本输入顺序以及输入特征子集，"
           "考察网络泛化性能的变化规律。")
    w.body("（4）设计实现简单的 BP 网络进行训练数据的识别分类。利用 Iris 鸢尾花数据集完成训练与测试，"
           "给出混淆矩阵、各类精确率/召回率/F1 值以及二维特征平面上的判决区域，并分析错分样本的成因。")

    w.h2("2. 实验目的")
    w.body("（1）掌握 BP 神经网络前向传播与误差反传的数学原理，能够脱离工具箱独立编程实现；")
    w.body("（2）掌握用控制变量法设计神经网络对比实验的方法，理解网络结构、超参数对收敛过程和泛化性能的影响；")
    w.body("（3）理解训练数据（样本量、量纲、噪声、类别分布、特征质量）对建模结果的影响，"
           "建立“数据决定上限、模型逼近上限”的认识；")
    w.body("（4）学会用混淆矩阵、精确率、召回率、F1 值等指标评价分类器，并能对实验结果作出机理解释。")

    w.h2("3. 实验要求")
    w.body("（1）BP 网络的前向传播与误差反传必须按公式自行编程实现，MATLAB 工具箱仅用于交叉验证；")
    w.body("（2）每组参数至少重复 5 个随机种子，结果以“均值±标准差”形式给出，"
           "避免把随机初始化带来的波动误判为参数的影响；")
    w.body("（3）训练集与测试集严格分离，归一化参数只能由训练集统计得到，防止信息泄漏；")
    w.body("（4）除记录数据外，须对实验现象给出机理解释。")


def build_section2(w, head):
    w.cursor = head

    w.h2("1. 硬件设备")
    w.body("实验在个人计算机上完成，主要配置为：PC 机（Windows 11 家庭版，Build 26200），"
           "Intel/AMD 多核处理器，8 GB 以上内存。")

    w.h2("2. 软件环境")
    w.table_caption("表 2-1  实验软硬件环境")
    w.table(
        ["项目", "名称", "版本/规格", "用途"],
        [["数值计算平台", "MATLAB", "R2024a（24.1）", "手写 BP 网络实现与全部实验"],
         ["神经网络工具箱", "Deep Learning Toolbox", "24.1", "仅用于 patternnet 交叉验证"],
         ["操作系统", "Windows 11 家庭版", "Build 26200", "实验平台"],
         ["数据集", "Iris 鸢尾花数据集", "150 样本 / 4 特征 / 3 类", "实验一参考数据"]],
        [2.6, 4.2, 4.4, 4.2], font_size=9)

    w.h2("3. 实验数据")
    w.body("数据由课程提供的三个制表符分隔文本文件构成，每行 5 个字段：前 4 列为花萼长、花萼宽、"
           "花瓣长、花瓣宽（单位 cm），第 5 列为类别标号 1/2/3，分别对应山鸢尾（setosa）、"
           "变色鸢尾（versicolor）和维吉尼亚鸢尾（virginica）。")
    w.table_caption("表 2-2  数据集划分情况")
    w.table(
        ["文件", "样本总数", "每类样本数", "用途"],
        [["trainData.txt", 75, "25 / 25 / 25", "训练集"],
         ["testData.txt", 75, "25 / 25 / 25", "测试集"],
         ["totalData.txt", 150, "50 / 50 / 50", "全集（备用）"]],
        [4.2, 3.0, 4.0, 4.0], font_size=9)


def build_section3(w, head):
    w.cursor = head

    w.h2("1. BP 网络结构与数学描述")
    w.body("实验采用三层前馈网络，输入层 4 个神经元对应 4 个特征，输出层 3 个神经元对应 3 个类别，"
           "隐层神经元个数可配置。隐层与输出层均取 logsig 激活函数，输出越接近 1 表示网络越倾向于判为该类，"
           "预测时取输出最大的节点作为类别。以 4-8-3 结构为例，其拓扑如图 3-1 所示。")
    w.picture("fig_bp_structure.png", 13.6)
    w.caption("图 3-1  BP 神经网络拓扑结构（4-8-3）")

    w.body("设第 l 层的权值矩阵为 W^(l)、偏置向量为 b^(l)，第 l-1 层输出为 a^(l-1)，则：")
    w.formula("前向传播：  z^(l) = W^(l)·a^(l-1) + b^(l)，  a^(l) = f( z^(l) )")
    w.formula("误差函数：  E = (1/N)·Σ(i=1..N) ½·Σ(j=1..3) ( t_ij − y_ij )²")
    w.formula("输出层误差：δ^(L) = ( y − t ) ⊙ f ′( z^(L) )")
    w.formula("隐层误差：  δ^(l) = ( W^(l+1) )ᵀ·δ^(l+1) ⊙ f ′( z^(l) )")
    w.formula("权值与偏置更新：  W^(l) ← W^(l) − η·δ^(l)·( a^(l-1) )ᵀ ，  b^(l) ← b^(l) − η·δ^(l)")
    w.body("式中 η 为学习率，⊙ 表示按元素相乘，N 为样本数。激活函数取 logsig 时 f′(x)=f(x)(1−f(x))；"
           "取 tansig 时 f′(x)=1−f(x)²；取 relu 时 f′(x)=1(x>0)。"
           "输出层若改用 softmax 并配合交叉熵损失，则输出层误差项简化为 δ^(L)=y−t。")

    w.body("在基本梯度下降的基础上，程序还实现了两种改进：一是附加动量项，"
           "ΔW(k)=mc·ΔW(k−1)+η·g(k)，利用历史梯度平滑更新方向；"
           "二是自适应学习率，当新的误差比上一步上升超过 4% 时回退该步并令 η←0.7η，"
           "误差下降则令 η←1.05η。二者结合即为实验中的 GDX 算法。")

    w.h2("2. 程序算法流程图")
    w.body("训练主流程如图 3-2 所示。图中“否”分支表示误差未达目标，继续迭代；"
           "达到目标误差、梯度足够小、验证集连续若干轮不再改善或达到最大迭代次数时终止训练。")
    w.picture("fig_flowchart.png", 9.0)
    w.caption("图 3-2  BP 算法训练流程图")

    w.h2("3. 程序文件清单")
    w.body("全部程序采用函数式编写，不使用任何神经网络训练函数。各文件功能见表 3-1。")
    w.table_caption("表 3-1  程序文件清单")
    w.table(
        ["文件名", "主要功能"],
        [["bp_init.m", "按指定层结构初始化 W、b，支持 Nguyen-Widrow、均匀、正态、全零四种方式"],
         ["bp_forward.m", "前向传播，返回各层净输入与输出"],
         ["bp_backward.m", "误差反传，计算各层梯度（支持 MSE 与交叉熵两种损失）"],
         ["bp_act.m / bp_dact.m", "激活函数及其导数（logsig、tansig、relu、purelin、softmax）"],
         ["bp_train.m", "训练主循环，支持 GD、GDM、GDA、GDX 与增量学习，含早停判据"],
         ["bp_loss.m / bp_predict.m", "误差评估与类别预测"],
         ["load_iris.m / onehot.m", "读取数据文件、标签 one-hot 编码"],
         ["norm_fit.m / norm_apply.m", "由训练集统计归一化参数并作用于任意数据"],
         ["train_eval.m / sweep_eval.m", "单次训练评估、批量单变量扫描"],
         ["run_exp1~5_*.m", "五个实验脚本，分别对应拓扑、参数、训练数据、分类、工具箱对比"],
         ["main.m", "一键运行全部实验，输出图片与结果表"]],
        [4.4, 10.0], font_size=9)


def build_section4(w, head):
    w.cursor = head

    hdr1, rows1 = read_xlsx("exp1_topology.xlsx")
    hdr2, rows2 = read_xlsx("exp2_params.xlsx")
    hdr3, rows3 = read_xlsx("exp3_data.xlsx")
    hdr4, rows4 = read_xlsx("exp4_class_metrics.xlsx")
    hdr5, rows5 = read_xlsx("exp4_errors.xlsx")
    hdr6, rows6 = read_xlsx("exp5_toolbox.xlsx")

    # ---------- 4.1
    w.h2("1. 数据读取与预处理")
    w.body("读入文本后用 readmatrix 按制表符分隔解析，剔除可能的空行，转置为 4×N 的矩阵，"
           "标签转换为 3×N 的 one-hot 矩阵。输入采用 z-score 标准化：")
    w.formula("x′ = ( x − μ ) / σ")
    w.body("其中 μ、σ 只由训练集统计得到，再作用于测试集，避免测试集信息泄漏。"
           "标准化后 4 个特征的均值约为 0、标准差约为 1，可显著改善 BP 网络的训练条件，"
           "这一点在 4.3 节和 4.5 节中有定量对比。")

    w.body("调试中遇到的主要问题是：函数 bp_loss 中最初写作 Y = bp_forward(net,X)，"
           "而该函数实际返回的是元胞数组，导致后续“cell 类型不支持运算符 −”的报错；"
           "改为 [A,~] = bp_forward(net,X); Y = A{end}; 后正常。"
           "另外发现 MATLAB 的变量名与字段名不支持中文，因此程序中统一使用英文标识符，"
           "仅注释、图表标题和数据表表头使用中文。")

    # ---------- 4.2
    w.h2("2. 手写 BP 算法的正确性验证")
    w.body("为确认误差反传公式实现无误，用中心差分法对解析梯度做数值校验："
           "在第 4-6-3 网络上取 10 个样本，对每个权值与偏置分别施加 ±10⁻⁶ 的扰动量，"
           "用 (E(w+h)−E(w−h))/(2h) 近似梯度并与反传结果比较，")
    w.body("得到 max|Δ(∂E/∂W)| = 5.54×10⁻¹¹，max|Δ(∂E/∂b)| = 2.85×10⁻¹¹，"
           "两者均在浮点误差量级，说明误差反传公式的实现是正确的。")

    # ---------- 4.3
    w.h2("3. 实验一：拓扑结构对训练结果的影响")
    w.body("固定训练算法 GDA（η=0.1）、Nguyen-Widrow 初始化与随机种子序列，"
           "只改变隐层数与隐层神经元个数。每个结构用 5 个随机种子重复，取均值。"
           "评价采用两种口径：(A) 固定 500 轮迭代预算，考察相同时代下的学习程度；"
           "(B) 训练到 MSE≤10⁻³ 或 2000 轮，考察收敛速度与最终泛化精度。"
           "同时设置标准化输入与原始输入两种条件，用以说明拓扑的影响与数据预处理是耦合的。")

    w.table_caption("表 4-1  不同拓扑结构的训练结果（5 个随机种子均值）")
    idx = {n: i for i, n in enumerate(hdr1)}
    d1 = {}
    for r in rows1:
        d1[(r[idx["拓扑结构"]], r[idx["输入条件"]])] = r
    order = [("4-3-3", "4-4-3", "4-5-3", "4-6-3", "4-8-3", "4-10-3", "4-12-3",
              "4-15-3", "4-20-3", "4-30-3", "4-50-3"),
             ("4-8-4-3", "4-10-6-3", "4-15-8-3", "4-20-10-3", "4-30-15-3")]
    trows = []
    for grp in order:
        for topo in grp:
            a = d1.get((topo, "zscore"))
            b = d1.get((topo, "none"))
            if a is None:
                continue
            trows.append([
                topo, fmt_i(a[idx["参数个数"]]),
                fmt_pct(a[idx["精度_500轮"]]), fmt_i(a[idx["达标迭代次数"]]),
                fmt_pct(a[idx["测试精度"]]),
                fmt_pct(b[idx["精度_500轮"]]), fmt_i(b[idx["达标迭代次数"]]),
                fmt_pct(b[idx["测试精度"]]),
            ])
    w.table(
        ["拓扑结构", "参数个数", "标准化 精度@500轮", "标准化 达标迭代", "标准化 测试精度",
         "原始 精度@500轮", "原始 达标迭代", "原始 测试精度"],
        trows, [1.9, 1.6, 2.1, 2.0, 1.9, 2.0, 1.9, 1.9], font_size=8.5, header_size=7.5)

    w.picture("exp1_topology_accuracy.png", 14.6)
    w.caption("图 4-1  500 轮预算下的测试精度随隐层规模的变化")
    w.picture("exp1_topology_epochs.png", 14.6)
    w.caption("图 4-2  收敛速度与相同时代预算下的拟合程度")
    w.picture("exp1_curves.png", 11.6)
    w.caption("图 4-3  不同拓扑结构的误差下降曲线（标准化输入）")

    w.body("从表 4-1 和图 4-1~4-3 可以看到以下现象：")
    w.body("（1）在标准化输入下，单隐层节点数由 3 增加到 50，达到 MSE=10⁻³ 的迭代次数仅由 1212 轮"
           "下降到 1025 轮，减少约 15%，边际收益很小；而双隐层结构只需 511~627 轮，约为单隐层的 50%，"
           "说明“加深“比”加宽”对加快收敛更有效。")
    w.body("（2）最终测试精度在 93.3%~95.5% 之间，标准差不超过 1 个测试样本；"
           "参数个数从 27 增加到 663（24 倍），精度并未随之提高。可见对本数据集而言，"
           "网络容量早已饱和，拓扑结构主要影响收敛速度，而不是最终泛化精度。")
    w.body("（3）原始输入（不归一化）时，16 个结构在 2000 轮内全部未能达到 MSE=10⁻³，"
           "500 轮后的训练 MSE 在 2.0×10⁻²~6.3×10⁻² 之间，比标准化时高约一个数量级；"
           "测试精度在 90.7%~96.5% 之间大幅跳变，结构之间不再有单调规律。"
           "这说明输入量纲不统一时，各权值的梯度尺度差异悬殊，训练效率和对结构的敏感性都显著恶化。")

    # ---------- 4.4
    w.h2("4. 实验二：各项参数对训练结果的影响")
    w.body("基准网络为 4-8-3，标准化输入，每次只改变一项参数，其余保持不变，"
           "每组 5 个随机种子取均值。训练统一采用 500 轮固定预算（第 7 组除外），"
           "这样可以观察“在有限迭代预算下”参数带来的差异。")

    w.table_caption("表 4-2  不同参数取值的训练结果（5 个随机种子均值）")
    j = {n: i for i, n in enumerate(hdr2)}
    trows = []
    for r in rows2:
        trows.append([
            r[j["参数组"]], r[j["取值"]], fmt_e(r[j["MSE"]]),
            fmt_pct(r[j["训练精度"]]),
            f'{fmt_pct(r[j["测试精度"]])}±{fmt_pct(r[j["测试精度_标准差"]])}',
            fmt_i(r[j["迭代次数"]]),
        ])
    w.table(["参数组", "取值", "末次 MSE", "训练精度/%", "测试精度/%", "迭代次数"],
            trows, [2.6, 3.6, 2.5, 2.2, 3.0, 1.8], font_size=8.5)

    w.picture("exp2_1_lr.png", 14.6)
    w.caption("图 4-4  学习率对误差下降过程的影响")
    w.picture("exp2_2_momentum.png", 14.6)
    w.caption("图 4-5  动量因子对误差下降过程的影响")
    w.picture("exp2_4_init.png", 14.6)
    w.caption("图 4-6  权值初始化方式的影响")
    w.picture("exp2_5_algorithm.png", 14.6)
    w.caption("图 4-7  四种训练算法的对比")
    w.picture("exp2_7_stability.png", 14.6)
    w.caption("图 4-8  线性输出层下的学习率稳定性")

    w.body("主要现象与分析：")
    w.body("（1）学习率是最敏感的参数。η 由 0.01 增至 2.0，500 轮后测试精度由 57.1% 单调升至 97.1%，"
           "训练 MSE 由 3.1×10⁻¹ 降至 1.6×10⁻²。在固定迭代预算下，小学习率对应的是“欠拟合“而非”稳定”。"
           "值得注意的是 η 增大到 2.0 仍未发散，原因是输出层 logsig 的导数 f(1−f) 最大仅为 0.25，"
           "误差项被限幅，等效步长有界。")
    w.body("（2）动量因子同样单调改善性能：mc=0 时 86.1%，mc=0.95 时 97.1%。"
           "动量项相当于对梯度做指数平滑，可抑制单一方向的振荡并加速平坦区域的推进。")
    w.body("（3）权值初始化中，全零初始化的网络完全失效，测试精度仅 32.0%（恰等于多数类占比），"
           "且第 73 轮就因梯度范数过小而停止。原因是同层所有神经元初始权值相同，"
           "误差反传时获得完全相同的梯度、始终保持相同的输出，网络等价于只有一个隐层神经元——"
           "这就是对称性问题。其余四种随机初始化的结果都在 94.7% 附近，差别不大。")
    w.body("（4）训练算法方面，同一 500 轮预算下 GD 仅 86.1%，GDM 达 96.8%，GDA 94.7%，GDX 93.6%。"
           "GDA/GDX 的训练 MSE 比 GD 低两个数量级（3.2×10⁻³ 与 1.8×10⁻¹），"
           "但测试精度反而略低于 GDM，说明训练误差更低并不必然带来更好的泛化。")
    w.body("（5）隐层激活函数的差异很小：logsig 94.7%、tansig 93.6%、relu 94.7%，"
           "相差不超过 1 个测试样本；输出层用 logsig+MSE 与用 softmax+交叉熵的结果也基本一致（94.7% 与 94.4%）。"
           "这是因为该数据集本身可分性好，网络容量远未成为瓶颈。")
    w.body("（6）把输出层改为线性（purelin）后，学习率的稳定性边界立刻显现：η=1 时仍可正常收敛（95.2%），"
           "η=2 时训练误差在第 1~2 轮即冲至 10²⁸² 量级并最终溢出为 Inf，η=5 时更早发散，"
           "分类退化为全部判为同一类（33.3%）。这与“输出层激活函数决定了误差曲面的局部曲率、"
           "进而决定稳定学习率上限”的结论一致。")

    # ---------- 4.5
    w.h2("5. 实验三：不同训练数据对建模的影响")
    w.body("固定 4-8-3 结构、GDA 算法（η=0.1）、2000 轮与 MSE=10⁻³ 的停止条件，"
           "只改变送入网络的训练数据，每个设置重复 5 个随机种子。")
    w.table_caption("表 4-3  不同训练数据设置下的建模结果")
    k = {n: i for i, n in enumerate(hdr3)}
    trows = []
    for r in rows3:
        rec = [fmt_pct(r[k["召回率_类1"]]), fmt_pct(r[k["召回率_类2"]]),
               fmt_pct(r[k["召回率_类3"]])]
        rec = ["—" if v == "—" else v for v in rec]
        trows.append([
            r[k["影响因素"]], r[k["设置"]],
            f'{fmt_pct(r[k["测试精度"]])}±{fmt_pct(r[k["测试精度_标准差"]])}',
            fmt_pct(r[k["训练精度"]]), "/".join(rec),
        ])
    w.table(["影响因素", "设置", "测试精度/%", "训练精度/%", "各类召回率 1/2/3"],
            trows, [2.7, 4.0, 3.2, 2.3, 3.4], font_size=8.5)

    w.picture("exp3_data_size_norm.png", 14.6)
    w.caption("图 4-9  训练样本数量与输入归一化方式的影响")
    w.picture("exp3_data_noise_feature.png", 14.6)
    w.caption("图 4-10  训练数据噪声与输入特征子集的影响")
    w.picture("exp3_data_imbalance_order.png", 14.6)
    w.caption("图 4-11  类别不均衡与样本输入顺序的影响")

    w.body("主要现象与分析：")
    w.body("（1）样本数量决定建模上限。每类仅 2 个样本（共 6 个）时测试精度只有 77.9%，"
           "每类 8 个（共 24 个）时升至 95.2%，此后继续增加到 75 个样本，精度稳定在 94% 左右甚至略有回落。"
           "值得注意的是，所有配置的训练精度都是 100%，也就是说小样本时的低精度完全来自泛化能力的不足，"
           "而不是拟合能力不足——网络已经“背下”了训练样本。")
    w.body("（2）归一化方式直接影响训练能否在预算内完成。不归一化和 min-max 到 [0,1] 时，"
           "2000 轮内均未达到目标误差（末次 MSE 分别为 9.8×10⁻³ 与 1.9×10⁻³）；"
           "z-score 只需 1073 轮即达标，末次 MSE 为 1.0×10⁻³。测试精度也依次为 91.7%、93.3%、93.9%。")
    w.body("（3）训练数据噪声是泛化的主要威胁。σ 由 0 增至 1.5（以标准差为单位），"
           "测试精度由 93.9% 单调降到 70.7%；而 σ≤0.2 时几乎无影响。"
           "同时训练精度直到 σ=0.6 仍保持 98% 以上，说明网络对噪声的“记忆”能力很强，"
           "训练误差指标无法反映噪声的危害。")
    w.body("（4）类别不均衡会使少数类被系统性牺牲。当训练集类别比例为 25/5/2 时，"
           "总体精度仍有 78.9%，但第 3 类召回率从均衡时的 85.6% 骤降到 51.2%，"
           "一半以上的第 3 类样本被判为第 2 类。原因是最小化均方误差时，"
           "少数类样本对总误差的贡献占比过低，网络倾向于把边界推向少数类一侧。")
    w.body("（5）样本输入顺序在增量学习（逐样本更新）下有明显影响：按类别排序训练 30 轮后测试精度 92.0%，"
           "随机打乱则达 94.4%，且误差下降更快。按类别排序时，网络在训练末段连续接收同一类样本，"
           "权值被最后一类“拉偏”，出现对后出现类别的偏好。批量训练时样本顺序无影响，"
           "因为每轮梯度是对全部样本求平均。")
    w.body("（6）特征质量比特征数量更重要。使用全部 4 维特征时测试精度 93.9%，"
           "只用花瓣长和花瓣宽两维反而升到 96.3%，而只用花萼两维时仅有 77.6%。"
           "花萼特征的类间重叠严重，属于“噪声维度”，引入后不仅无益，还增加了需要估计的参数。")

    # ---------- 4.6
    w.h2("6. 实验四：Iris 数据集的识别分类")
    w.body("采用 4-8-3 结构、GDA 算法（η=0.1）、2000 轮与 MSE=10⁻³ 停止条件，"
           "在 trainData（75 个样本）上训练，在 testData（75 个样本）上测试。"
           "为考察随机初始化的影响，用 10 个不同随机种子重复训练，"
           "得到测试精度 93.60%±0.56%，最好一次为 94.67%（对应 75 个测试样本中错 4 个）。")

    w.table_caption("表 4-4  测试集混淆矩阵（行=真实类别，列=预测类别）")
    w.table(["真实 \\ 预测", "类1 山鸢尾", "类2 变色鸢尾", "类3 维吉尼亚"],
            [["类1 山鸢尾", 25, 0, 0],
             ["类2 变色鸢尾", 0, 24, 1],
             ["类3 维吉尼亚", 0, 3, 22]],
            [3.6, 3.4, 3.4, 3.4], font_size=9)

    w.table_caption("表 4-5  各类别的精确率、召回率与 F1 值")
    trows = []
    names = {1: "类1 山鸢尾", 2: "类2 变色鸢尾", 3: "类3 维吉尼亚"}
    for r in rows4:
        trows.append([names.get(r[0], r[0]), fmt_pct(r[1], 2), fmt_pct(r[2], 2), fmt_pct(r[3], 2)])
    w.table(["类别", "精确率/%", "召回率/%", "F1/%"], trows,
            [4.2, 3.4, 3.4, 3.4], font_size=9)

    w.body("宏平均精确率为 94.85%、召回率 94.67%、F1 值 94.66%。"
           "第 1 类（山鸢尾）线性可分，识别率达到 100%；"
           "错误全部发生在第 2 类与第 3 类之间，这与两类在花瓣尺寸上存在重叠区的事实相符。")

    w.picture("exp4_confusion.png", 14.6)
    w.caption("图 4-12  测试集混淆矩阵与各类识别指标")

    w.table_caption("表 4-6  被分错的测试样本")
    trows = []
    for r in rows5:
        trows.append([r[1], f"类{r[2]}", f"类{r[3]}",
                      f"{r[4]:.1f}", f"{r[5]:.1f}", f"{r[6]:.1f}", f"{r[7]:.1f}"])
    w.table(["测试集样本号", "真实类别", "预测类别", "花萼长", "花萼宽", "花瓣长", "花瓣宽"],
            trows, [2.6, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], font_size=9)

    w.body("4 个错分样本的花瓣长都在 4.8~5.8 cm、花瓣宽在 1.5~1.8 cm 之间，"
           "正好落在两类交界的过渡带；其中 3 个真实类别为第 3 类却被判为第 2 类，"
           "说明判决边界偏向第 3 类一侧，网络对第 3 类的召回率（88.0%）低于第 2 类（96.0%）。"
           "这与 4.5 节中类别数量越少、召回率越低的规律一致。")

    w.picture("exp4_decision_boundary.png", 12.4)
    w.caption("图 4-13  仅用花瓣两维输入时网络在特征平面上的判决区域")

    w.body("为进一步直观展示网络学到的分类边界，另建一个只以花瓣长、花瓣宽为输入的 2-8-3 网络。"
           "该网络训练精度与测试精度均为 96.0%，反而高于 4 维网络。"
           "从图 4-13 可见，判决边界由两段近似平行的直线构成，"
           "第 2 类与第 3 类之间是一条斜线，正好穿过两类的重叠区。"
           "这说明 BP 网络在该数据上学到的本质上是分段线性的判别函数，"
           "与线性判别分析的结果接近，也再次印证了增加隐层节点并不能突破数据本身的重叠限制。")

    # ---------- 4.7
    w.h2("7. 实验五：与 MATLAB 工具箱的交叉验证")
    w.body("为验证手写实现的正确性，用 Deep Learning Toolbox 的 patternnet 建立同样 8 个隐层神经元的网络，"
           "使用相同的训练/测试数据与归一化方式，分别采用 trainscg、trainlm、traingd 三种算法训练，"
           "并关闭自动划分验证集（divideFcn='dividetrain'），使其与手写网络使用完全相同的训练样本。")

    w.table_caption("表 4-7  手写实现与工具箱训练结果对比")
    trows = []
    for r in rows6:
        trows.append([r[0], f"{float(r[1]):.2f}", f"{float(r[2]):.2f}",
                      fmt_i(r[3]), f"{r[4]:.2f}"])
    w.table(["方法", "训练精度/%", "测试精度/%", "迭代次数", "耗时/s"],
            trows, [6.4, 3.0, 3.0, 2.4, 2.2], font_size=9)

    w.picture("exp5_toolbox_compare.png", 14.6)
    w.caption("图 4-14  手写实现与工具箱的精度、耗时对比")

    w.body("手写网络的测试精度为 94.67%，工具箱三种算法为 92.0%~93.33%，"
           "两者相差不到 2 个测试样本，且手写网络训练精度同为 100%，"
           "可以认为手写实现在功能与精度上与工具箱一致，算法实现正确。"
           "在迭代次数上差别很大：手写 GDA 用了 1090 轮，而 trainscg 仅 14 轮、trainlm 仅 8 轮，"
           "这是因为共轭梯度法与 Levenberg-Marquardt 法利用了二阶信息，"
           "在中小规模网络上通常比一阶梯度法快一到两个数量级。"
           "但单次迭代耗时也相应更长，在本问题的规模下绝对时间差别不大。")


    # ---------- 4.8 按课堂方法补充的验证 ----------
    w.h2("8. 按课堂方法补充的验证：ROC/AUC 与 K 折交叉验证")
    w.body("课堂第 6 节「感知器的验证」给出了检验分类器的规范做法：训练集与测试集必须分开"
           "（训练占 70%~80%、测试占 20%~30%，并强调用训练集自评会产生“过拟合假象”）；"
           "用 K 折交叉验证降低单次划分的偶然性；用混淆矩阵及准确率、精确率、召回率、F1 "
           "四项指标评价；用 ROC 曲线与 AUC 衡量全阈值范围内的区分能力。"
           "4.6 节已给出混淆矩阵与四项指标，本节补做 ROC/AUC 与 5 折交叉验证。")

    hdr7, rows7 = read_xlsx("exp6_auc.xlsx")
    w.body("先看 ROC 与 AUC。三分类问题按“一对其余”拆成三个二分类问题：以第 c 类的输出值作为"
           "“属于第 c 类”的置信分数，其余两类合并为负类，逐个切换判决阈值就得到一条 ROC 曲线。"
           "AUC 是曲线下面积，取值 0.5（等同于随机猜测）到 1（完全可分）。")
    w.table_caption("表 4-8  各类别的一对其余 AUC")
    j7 = {n: i for i, n in enumerate(hdr7)}
    trows = []
    for r in rows7:
        trows.append([r[j7["名称"]], f"{float(r[j7['AUC']]):.4f}"])
    w.table(["类别", "AUC"], trows, [7.0, 4.0], font_size=9)

    w.picture("exp4_roc_auc.png", 10.6)
    w.caption("图 4-15  测试集上的一对其余 ROC 曲线")
    w.body("三个类别的 AUC 分别为 1.0000、0.9968、0.9960，宏平均 0.9976，三条曲线都紧贴左上角。"
           "第 1 类的 AUC 达到 1，与 4.6 节中该类 100% 的准确率一致；第 2、3 类的 AUC 略低于 1，"
           "对应两类交界处那几个被反复误判的样本。AUC 接近 1 说明网络不仅“判对得多”，"
           "而且“判对的把握大”，改变判决阈值结果也稳定。")

    w.body("再看 K 折交叉验证。把 totalData 的 150 条样本随机分成 5 个互斥子集，每次用其中 4 个训练、"
           "剩下 1 个测试，轮流 5 次后取平均。这样每条样本都恰好被测试一次，可避免单次划分的偶然性。"
           "每一折的归一化参数仍然只在当折的训练集上统计。")
    hdr8, rows8 = read_xlsx("exp6_cv5.xlsx")
    w.table_caption("表 4-9  5 折交叉验证结果")
    j8 = {n: i for i, n in enumerate(hdr8)}
    trows = []
    for r in rows8:
        v = r[j8["折次"]]
        label = "平均" if (v is None or v != v) else f"第{int(v)}折"
        trows.append([label, int(r[j8["测试样本数"]]),
                      f"{100*float(r[j8['测试精度']]):.2f}",
                      f"{float(r[j8['宏平均AUC']]):.4f}"])
    w.table(["折次", "测试样本数", "测试精度/%", "宏平均 AUC"], trows,
            [3.2, 3.4, 3.6, 3.6], font_size=9)

    w.body("5 折平均精度为 96.00%±2.79%，宏平均 AUC 为 0.9938±0.0074，"
           "与 4.6 节固定测试集上的 93.60%±0.56% 处于同一水平，说明前面的结论不是某一次划分的偶然结果。"
           "交叉验证的标准差反而更大（2.79% 对 0.56%），是因为每折只有 30 个测试样本，"
           "一个样本就相当于 3.3 个百分点——这正是课堂强调“单次划分有偶然性、要用 K 折取平均”的原因。")


def build_section5(w, head):
    w.cursor = head

    w.h2("1. 主要结论")
    w.body("（1）拓扑结构主要决定收敛速度，而非最终精度。在本数据集上，“双隐层”把达到目标误差的迭代次数"
           "从约 1100 轮降到约 500 轮（约 2 倍加速），但把单隐层节点数从 3 加到 50、"
           "参数个数增加 15 倍，测试精度始终停留在 93%~95%，说明精度瓶颈不在网络容量而在数据本身的可分性。")
    w.body("（2）数据预处理是 BP 网络可用的前提。不做归一化时，2000 轮内没有任何一个结构达到目标误差，"
           "训练 MSE 比标准化时高约一个数量级，且结果对结构高度敏感。"
           "原因是输入量纲不统一会使不同权重的梯度尺度差异悬殊，梯度下降在病态曲面上难以推进。")
    w.body("（3）参数的影响必须放在“迭代预算”的语境下讨论。固定 500 轮时，学习率越大越好（0.01→2.0，精度 57%→97%），"
           "动量越大越好（0→0.95，精度 86%→97%）。这并不是说学习率可以无限增大："
           "当输出层改为线性（purelin）时，η=2 立刻发散。"
           "所谓“最优学习率”其实是误差曲面曲率与迭代预算共同决定的。")
    w.body("（4）权值初始化必须打破对称性。全零初始化让同层所有神经元永远保持相同输出，"
           "网络退化为单神经元结构，测试精度只有 32%，是所有参数中后果最严重、也最容易被忽视的一个。")
    w.body("（5）训练误差低不等于泛化好。实验二中 GDA、GDX 的训练 MSE 比 GDM 低一到两个数量级，"
           "但测试精度反而略低；实验三中噪声 σ=0.6 时训练精度仍有 98%，测试精度却已跌到 89%。"
           "评价模型必须使用独立测试集，而不能只看训练误差。")
    w.body("（6）训练数据的数量、质量与分布共同决定泛化上限。样本量低于每类 8 个时精度迅速下降；"
           "噪声 σ>0.6 后精度快速劣化；类别比例 25/5/2 时少数类召回率跌破 52%；"
           "而用花瓣两维特征反而优于全部 4 维，说明与任务无关的特征维度是负担而不是信息。")
    w.body("（7）手写实现与工具箱结果一致。手写网络的测试精度 94.67% 与工具箱 92.0%~93.33% 处于同一水平，"
           "数值梯度校验误差在 10⁻¹¹ 量级，证明前向传播与误差反传的公式实现正确。")

    w.h2("2. 对实验现象的进一步分析")
    w.body("为什么拓扑结构对精度影响很小、对收敛速度影响却很大？从优化角度看，"
           "增加隐层神经元或增加层数，本质上是缩短了从初始点到极小点的“路径”，"
           "使每一步的下降方向更少出现锯齿，因此收敛更快；"
           "但代价函数的最小值是由数据分布决定的，网络容量一旦超过拟合数据所需，"
           "继续增加参数只是让最小值附近存在更多等价的解，并不改变测试误差的下界。"
           "Iris 数据集中第 2、3 类在花瓣尺寸上存在真实重叠，"
           "任何判别函数在这片区域都只能“二选一”，这才是精度卡在 94%~96% 的根本原因。")
    w.body("为什么全零初始化会失效？BP 网络的隐层节点之间是对称的："
           "若同一层的所有权值相同，则每个节点对同一输入产生相同输出、"
           "在反传时得到相同梯度、更新后仍然相同。这一对称性一旦形成就无法自行打破，"
           "网络的有效容量退化为 1 个隐层节点。随机初始化的作用正是在参数空间中“错开”各个节点，"
           "这也是 Nguyen-Widrow 一类初始化方法在随机之外还要按扇入扇出做尺度归一化的原因。")
    w.body("为什么类别不均衡会牺牲少数类？以均方误差为目标时，每个样本对总误差的贡献相同，"
           "而少数类样本数量少，其误差在总误差中占比低，梯度被多数类“淹没”。"
           "网络把判决边界推向少数类一侧，可以用很小的少数类误判换取多数类误差的进一步下降——"
           "从总误差角度看是“最优”的，从分类任务角度看却是灾难性的。"
           "改进办法包括重采样、类别加权损失或按类别调整判决阈值。")

    w.h2("3. 实验的不足与改进方向")
    w.body("（1）训练算法只实现了一阶方法（GD、动量、自适应学习率），"
           "与工具箱的 trainscg、trainlm 相比迭代次数高出一到两个数量级。"
           "后续可补充共轭梯度法或 Levenberg-Marquardt 法，以便在迭代效率上做更公平的比较。")
    w.body("（2）未引入验证集做早停。当前以固定的目标误差或迭代上限作为停止条件，"
           "在噪声较大的设置下容易过拟合。引入验证集早停可以更客观地反映泛化性能。")
    w.body("（3）分类精度的统计分辨率有限。测试集只有 75 个样本，一个样本就对应 1.33 个百分点，"
           "因此许多参数之间的差异（如三种激活函数）落在统计噪声范围内。"
           "更严谨的做法是采用 k 折交叉验证，或在 totalData 上多次随机划分并统计置信区间。")
    w.body("（4）只使用了单个数据集。Iris 的可分性很好，导致许多“应该出现”的现象（如明显的过拟合）"
           "被数据本身的简单性掩盖。若能补充一个线性不可分的数据集（如双月牙或带噪的异或问题），"
           "拓扑结构与网络容量的影响会表现得更充分。")

    w.h2("4. 实验心得")
    w.body("这次实验最大的收获是“把公式变成代码“的过程。此前对 BP 算法的理解停留在"
           "”误差反传就是链式法则”这一句话上，真正动手写时才发现，"
           "输出层误差项要不要乘 f′(z)、隐层误差里 W 的转置该放在哪一边、"
           "批量与增量的梯度要不要除以 N，这些细节只要有一处写错，程序仍能运行、误差也会下降，"
           "但收敛速度和最终精度会莫名其妙地变差。用中心差分做数值梯度校验，"
           "把这 10⁻¹¹ 量级的误差摆在眼前，才真正确认实现是对的——这也让我体会到"
           "“可验证“比“能跑通”重要得多。")
    w.body("第二个收获是对”实验设计”的理解。最初我按直觉比较不同拓扑结构的测试精度，"
           "发现几乎都是 94%~95%，差异完全淹没在随机初始化的波动里，一度以为实验失败了。"
           "后来改为固定随机种子序列、每个配置重复 5 次取均值±标准差，并把评价口径分成"
           "“相同迭代预算下的精度”和”达到目标误差所需迭代次数”两种，规律才显现出来："
           "拓扑结构其实强烈影响着收敛速度，只是不影响最终精度。"
           "这让我明白，控制变量和统计重复不是形式上的要求，而是让现象“显形”的必要手段。")
    w.body("第三个收获是对数据与模型关系的重新认识。实验三中，每类 2 个样本时精度只有 77.9%，"
           "而把 4 维特征换成花瓣 2 维反而从 93.9% 升到 96.3%；"
           "类别比例变成 25/5/2 后，少数类召回率直接腰斩。"
           "这些结果比任何一句“数据决定上限”都更有说服力："
           "在网络容量已经足够的情况下，继续调参的收益远小于清洗数据、选对特征、"
           "处理类别不均衡带来的收益。")
    w.body("最后，把工具箱的结果作为“标准答案”来交叉验证自己的实现，也是一种有效的学习方法。"
           "当看到手写网络 94.67% 与 trainlm 93.33% 只差 1 个样本时，"
           "那种“自己写的算法真的能用”的确认感，是单纯调用 newff 几行代码所给不了的。")


if __name__ == "__main__":
    main()
