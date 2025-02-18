c.ContentsManager.allow_hidden = True

# For Matplotlib inline plot with the mplcairo backend
import mplcairo.base
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.new_figure_manager = mplcairo.base.new_figure_manager
