"""
Reference: https://github.com/matplotlib/matplotlib/issues/22297#issue-1112035885

PGF Text

Render text in LaTeX using PGF, everything else is rendered as a separate PDF
that is then included in the PGF.
"""

import codecs
import os

from matplotlib import _api, cbook
from matplotlib.backend_bases import (
     _Backend, FigureCanvasBase, RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import RendererPdf, PdfFile
from matplotlib.backends.backend_pgf import RendererPgf, get_preamble, get_fontspec, writeln

class RendererPgfText(RendererBase):
    def __init__(self, figure, dpi, pgffh, pdffile):
        super().__init__()

        width, height = figure.get_size_inches()
        self.dpi = dpi

        self.pdf_renderer = None
        self.pgf_renderer = None

        if pgffh is not None:
          self.pgf_renderer = RendererPgf(figure, pgffh)
        if pdffile is not None:
          self.pdf_renderer = RendererPdf(pdffile, self.dpi, height, width)

        either = [
          'get_text_width_height_descent',
          'get_canvas_width_height',
          'points_to_pixels',
          'new_gc',
          'flipy',
        ]

        pdf_only = [
          'finalize',
          'draw_path',
          'draw_markers',
          'draw_path_collection',
          'draw_quad_mesh',
          'draw_image',
        ]

        pgf_only = [
          'draw_text',
          'draw_tex',
        ]

        for fn in either:
          self.forward_f(fn)

        for fn in pgf_only:
          self.forward_excl(self.pgf_renderer, fn)

        for fn in pdf_only:
          self.forward_excl(self.pdf_renderer, fn)

    def forward_excl(self, renderer, fname):
      def f(*args, **kwargs):
        if renderer is not None:
          return getattr(renderer, fname)(*args, **kwargs)

      setattr(self, fname, f)

    def forward_f(self, fname):
      def f(*args, **kwargs):
        if self.pgf_renderer:
          return getattr(self.pgf_renderer, fname)(*args, **kwargs)
        else:
          return getattr(self.pdf_renderer, fname)(*args, **kwargs)

      setattr(self, fname, f)

class FigureCanvasPgfText(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc.

    Note: GUI templates will want to connect events for button presses,
    mouse movements and key presses to functions that call the base
    class methods button_press_event, button_release_event,
    motion_notify_event, key_press_event, and key_release_event.  See the
    implementations of the interactive backends for examples.

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance
    """

    # You should provide a print_xxx function for every file format
    # you can write.

    # If the file type is not in the base set of filetypes,
    # you should add it to the class-scope filetypes dictionary as follows:
    filetypes = {**FigureCanvasBase.filetypes, 'pgf': 'My magic pgf TEXT format'}

    @_api.delete_parameter("3.5", "args")
    def print_pgf(self, filename, metadata=None, *args, **kwargs):
        """
        Write out format pgf.

        This method is normally called via `.Figure.savefig` and
        `.FigureCanvasBase.print_figure`, which take care of setting the figure
        facecolor, edgecolor, and dpi to the desired output values, and will
        restore them to the original values.  Therefore, `print_foo` does not
        need to handle these settings.
        """

        header_text = """%% Creator: Matplotlib, PGF Text backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Also ensure that all the required font packages are loaded; for instance,
%% the lmodern package is sometimes necessary when using math font.
%%   \\usepackage{lmodern}
%%
%% Figures using additional raster images (or non text) can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%%
%% and then include the figures with
%%   \\import{<path to file>}{<filename>.pgf}
%%
"""
        # append the preamble used by the backend as a comment for debugging
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        for line in get_fontspec().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # get figure size in inch
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.get_dpi()
        if isinstance(filename, str):
          pdfname = filename[:-4] + '.pdf'
        else:
          pdfname = ""

        # create pgfpicture environment and write the pgf code
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            if not cbook.file_requires_unicode(fh):
                fh = codecs.getwriter("utf-8")(fh)

            fh.write(header_text)
            fh.write(header_info_preamble)
            fh.write("\n")
            writeln(fh, r"\begingroup")
            writeln(fh, r"\makeatletter")
            writeln(fh, r"\begin{pgfpicture}")
            writeln(fh,
                    r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                    % (w, h))
            writeln(fh, r"\pgfusepath{use as bounding box, clip}")
            writeln(fh, r"\begin{pgfscope}\includegraphics{" + os.path.basename(pdfname) + r"}\end{pgfscope}")

            renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                         RendererPgfText(self.figure, dpi, pgffh=fh, pdffile=None))
            self.figure.draw(renderer)
            renderer.finalize()

            # end the pgfpicture environment
            writeln(fh, r"\end{pgfpicture}")
            writeln(fh, r"\makeatother")
            writeln(fh, r"\endgroup")

        # Need to somehow do this while still "rendering" pgf at default dpi of 100 because that's what the original DPI uses...
        self.figure.set_dpi(72)            # there are 72 pdf points to an inch
        pdffile = PdfFile(pdfname, metadata=metadata)
        pdffile.newPage(w, h)
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     RendererPgfText(self.figure, dpi, pgffh=None, pdffile=pdffile))
        self.figure.draw(renderer)
        renderer.finalize()
        pdffile.finalize()
        pdffile.close()

    def get_default_filetype(self):
        return 'pgf'

@_Backend.export
class _BackendPgfText(_Backend):
    FigureCanvas = FigureCanvasPgfText
