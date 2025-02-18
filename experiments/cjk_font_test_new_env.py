# -*- coding: utf-8 -*-


import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


FIG_SIZE = (8,4.5)
DPI = 120
CHINESE_FP = fm.FontProperties(fname='./font/source_han_serif_sc.ttf')
# CHINESE_FP = fm.FontProperties(fname='./font/source_han_serif_sc.ttf', math_fontfamily="stix")
# CHINESE_FP = fm.FontProperties(fname='/home/derek/Documents/gleap/gleap-journal/experiments/font/source_han_serif_sc.ttf', math_fontfamily="stix")
# CHINESE_FP = fm.FontProperties(fname='./font/SourceHanSerifSC-VF.ttf', math_fontfamily="stix")
# CHINESE_FP = fm.FontProperties(fname='./font/source_han_serif_sc_regular.otf', math_fontfamily="stix")


fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
ax.set_xlabel(r'测试中文字体 normal', fontsize=32.0, fontweight="normal", fontproperties=CHINESE_FP)
ax.set_ylabel(r'测试中文字体 bold', fontsize=32.0, fontweight="bold", fontproperties=CHINESE_FP)
# Save the figure
fig.savefig('./cjk_font_test_new_env.pdf', format='pdf', bbox_inches='tight')
fig.savefig('./cjk_font_test_new_env.png', format='png', bbox_inches='tight')
