"""
LaTeX template for calculation packages.

Jinja2 template with custom delimiters to avoid LaTeX brace conflicts.
Produces a .tex document compiled to PDF via pdflatex.

Delimiter convention (standard for Jinja2 + LaTeX):
  Block tags:    \\BLOCK{...}
  Variable tags: \\VAR{...}
  Comment tags:  \\#{...}
"""

LATEX_TEMPLATE = r"""
\documentclass[11pt,letterpaper]{article}

% ---- Layout ----
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}

% ---- Typography ----
\usepackage[T1]{fontenc}
\usepackage{mathpazo}
\usepackage{inconsolata}

% ---- Math ----
\usepackage{amsmath}
\usepackage{amssymb}

% ---- Tables ----
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}

% ---- Graphics ----
\usepackage{graphicx}
\usepackage{float}

% ---- Color ----
\usepackage[dvipsnames]{xcolor}

% ---- Misc ----
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\usepackage{mdframed}
\usepackage{calc}

% ---- Custom colors ----
\definecolor{StepBlue}{HTML}{2563EB}
\definecolor{StepBlueDark}{HTML}{1E40AF}
\definecolor{PassGreen}{HTML}{16A34A}
\definecolor{FailRed}{HTML}{DC2626}
\definecolor{StepBg}{HTML}{FAFAFA}
\definecolor{RefGray}{HTML}{888888}
\definecolor{SubGray}{HTML}{555555}

% ---- Calc step box ----
\mdfdefinestyle{calcstep}{%
  leftmargin=0pt,
  rightmargin=0pt,
  innerleftmargin=12pt,
  innerrightmargin=12pt,
  innertopmargin=8pt,
  innerbottommargin=8pt,
  backgroundcolor=StepBg,
  linewidth=0pt,
  leftline=true,
  linecolor=StepBlue,
  innerleftmargin=14pt,
}

% ---- Pass/fail boxes ----
\mdfdefinestyle{passbox}{%
  leftmargin=0pt, rightmargin=0pt,
  innerleftmargin=12pt, innerrightmargin=12pt,
  innertopmargin=6pt, innerbottommargin=6pt,
  backgroundcolor=PassGreen!8,
  linewidth=0pt, leftline=true,
  linecolor=PassGreen, linewidth=3pt,
}

\mdfdefinestyle{failbox}{%
  leftmargin=0pt, rightmargin=0pt,
  innerleftmargin=12pt, innerrightmargin=12pt,
  innertopmargin=6pt, innerbottommargin=6pt,
  backgroundcolor=FailRed!8,
  linewidth=0pt, leftline=true,
  linecolor=FailRed, linewidth=3pt,
}

% ---- Section formatting ----
\titleformat{\section}{\large\bfseries\MakeUppercase}{}{0em}{}[\vspace{-4pt}\rule{\textwidth}{1pt}\vspace{4pt}]
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}

% ---- Suppress page numbers in favor of fancyhdr ----
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\footnotesize\thepage}

% ---- Tight list ----
\setlist{nosep, leftmargin=1.5em}

\begin{document}

% ======== TITLE BLOCK ========
\begin{center}
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\vspace{4pt}
{\Large\bfseries\MakeUppercase{\VAR{title}}} \\[8pt]
\begin{tabular}{@{}ll@{\hspace{2cm}}ll@{}}
\textbf{Project:} & \VAR{project_name} &
\textbf{Number:} & \VAR{project_number} \\[2pt]
\textbf{Prepared By:} & \VAR{engineer} &
\textbf{Date:} & \VAR{date} \\[2pt]
\BLOCK{if checker}
\textbf{Checked By:} & \VAR{checker} &
\BLOCK{endif}
\BLOCK{if company}
\textbf{Company:} & \VAR{company} \\
\BLOCK{endif}
\end{tabular}
\vspace{4pt} \\
\hline
\end{tabular}
\end{center}

\vspace{12pt}

% ======== SECTIONS ========
\BLOCK{for section in sections}
\section*{\VAR{section_number}.~\VAR{section.title}}
\addcontentsline{toc}{section}{\VAR{section_number}.~\VAR{section.title}}

\BLOCK{for item in section.items}
\BLOCK{if item._type == "input_table"}
% ---- Input parameter table ----
\begin{longtable}{p{0.35\textwidth} >{\raggedright}p{0.15\textwidth} >{\raggedleft}p{0.20\textwidth} p{0.15\textwidth}}
\toprule
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} & \textbf{Unit} \\
\midrule
\BLOCK{for row in item.rows}
\VAR{row[0]} & $\VAR{row[1]}$ & \texttt{\VAR{row[2]}} & \VAR{row[3]} \\
\BLOCK{endfor}
\bottomrule
\end{longtable}

\BLOCK{elif item._type == "calc_step"}
% ---- Calculation step ----
\begin{mdframed}[style=calcstep]
{\color{StepBlueDark}\textbf{\VAR{item.title}}}

\BLOCK{if item.equation}
\vspace{4pt}
\begin{equation*}
\VAR{item.equation_latex}
\end{equation*}
\BLOCK{endif}

\BLOCK{if item.substitution}
{\color{SubGray}\begin{equation*}
\VAR{item.substitution_latex}
\end{equation*}}
\BLOCK{endif}

\vspace{2pt}
\noindent\hspace{1em}$\VAR{item.result_name_latex} = \textcolor{StepBlue}{\mathbf{\VAR{item.result_value}}}$ \VAR{item.result_unit}

\BLOCK{if item.reference}
\vspace{2pt}
{\footnotesize\color{RefGray}\textit{\VAR{item.reference}}}
\BLOCK{endif}

\BLOCK{if item.notes}
\vspace{2pt}
{\footnotesize\color{SubGray}\VAR{item.notes}}
\BLOCK{endif}
\end{mdframed}
\vspace{4pt}

\BLOCK{elif item._type == "check"}
% ---- Engineering check ----
\BLOCK{if item.passes}
\begin{mdframed}[style=passbox]
\textcolor{PassGreen}{\textbf{PASS}} \quad \VAR{item.description}
\end{mdframed}
\BLOCK{else}
\begin{mdframed}[style=failbox]
\textcolor{FailRed}{\textbf{FAIL}} \quad \VAR{item.description}
\end{mdframed}
\BLOCK{endif}
\noindent\hspace{2em}{\small\color{SubGray}\texttt{%
$\VAR{item.demand_label_latex}$ = \VAR{item.demand} \VAR{item.unit}%
\quad \VAR{item.relation} \quad%
$\VAR{item.capacity_label_latex}$ = \VAR{item.capacity} \VAR{item.unit}%
\quad (D/C = \VAR{item.dc_ratio})%
}}
\vspace{8pt}

\BLOCK{elif item._type == "figure"}
% ---- Figure ----
\begin{figure}[H]
\centering
\includegraphics[width=\VAR{item.width_frac}\textwidth]{\VAR{item.image_path}}
\BLOCK{if item.caption}
\caption*{\VAR{item.caption}}
\BLOCK{endif}
\end{figure}

\BLOCK{elif item._type == "table"}
% ---- Data table ----
\BLOCK{if item.title}
\noindent\textbf{\VAR{item.title}}
\vspace{2pt}
\BLOCK{endif}

\begin{longtable}{\VAR{item.col_spec}}
\toprule
\BLOCK{for h in item.headers}
\textbf{\VAR{h}} \BLOCK{if not loop.last} & \BLOCK{endif}
\BLOCK{endfor} \\
\midrule
\BLOCK{for row in item.rows}
\BLOCK{for cell in row}
\VAR{cell} \BLOCK{if not loop.last} & \BLOCK{endif}
\BLOCK{endfor} \\
\BLOCK{endfor}
\bottomrule
\end{longtable}
\BLOCK{if item.notes}
{\footnotesize\color{SubGray}\textit{\VAR{item.notes}}}
\BLOCK{endif}

\BLOCK{elif item._type == "text"}
% ---- Text paragraph ----
\VAR{item.text}

\vspace{4pt}
\BLOCK{endif}
\BLOCK{endfor}

\BLOCK{set section_number = section_number + 1}
\BLOCK{endfor}

% ======== REFERENCES ========
\BLOCK{if references}
\section*{\VAR{section_number}.~References}

\begin{enumerate}
\BLOCK{for ref in references}
\item \VAR{ref}
\BLOCK{endfor}
\end{enumerate}
\BLOCK{endif}

% ======== FOOTER ========
\vfill
\noindent\rule{\textwidth}{0.4pt}
\begin{center}
{\footnotesize Generated by GeotechStaffEngineer --- \VAR{date}}
\end{center}

\end{document}
"""
