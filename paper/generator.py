"""
PaperGenerator — takes LaTeX from the LLM and compiles it to PDF.
Handles figure injection, template wrapping if needed, and pdflatex compilation.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Fallback if template file is missing
_FALLBACK_WRAPPER = r"""
\documentclass[12pt]{{article}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{booktabs}}
\usepackage{{listings}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\title{{{title}}}
\author{{{author}}}
\date{{\today}}

\begin{{document}}
\maketitle

{body}

\end{{document}}
"""


class PaperGenerator:
    def __init__(self, config: dict):
        self.cfg = config["paper"]
        self.output_dir = Path(self.cfg.get("output_dir", "papers"))
        self.compiler = self.cfg.get("compiler", "pdflatex")
        self.template_name = self.cfg.get("template", "plain")
        self.author = self.cfg.get("authors", "ML Research Agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_wrapper(self) -> str:
        """Load the configured LaTeX wrapper template, fall back to built-in if missing."""
        tpl_path = TEMPLATES_DIR / f"{self.template_name}.tex"
        if tpl_path.exists():
            return tpl_path.read_text(encoding="utf-8")
        logger.warning(
            f"Template '{self.template_name}.tex' not found in {TEMPLATES_DIR} "
            f"— using built-in fallback"
        )
        return _FALLBACK_WRAPPER

    def compile(self, latex: str, title: str, artifacts: dict[str, str] = None) -> Path:
        """
        Compile LaTeX string to PDF. Returns path to the PDF.
        artifacts: {filename: local_path} for images/data to include
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Ensure it's a complete document
            if r"\begin{document}" not in latex:
                logger.warning(
                    f"LaTeX is not a full document — wrapping with '{self.template_name}' template"
                )
                wrapper = self._load_wrapper()
                latex = wrapper.format(title=title, author=self.author, body=latex)

            # Copy artifact files into build dir
            if artifacts:
                results_dir = tmpdir / "results"
                results_dir.mkdir()
                for fname, local_path in artifacts.items():
                    src = Path(local_path)
                    if src.exists():
                        shutil.copy(src, results_dir / fname)
                        logger.debug(f"Copied artifact: {fname}")

            # Write main.tex
            main_tex = tmpdir / "main.tex"
            main_tex.write_text(latex, encoding="utf-8")

            # Run compiler (twice for references/TOC)
            pdf_path = self._run_compiler(tmpdir, main_tex)

            slug = self._slugify(title)
            dest_tex = self.output_dir / f"{slug}.tex"
            shutil.copy(main_tex, dest_tex)

            if pdf_path and pdf_path.exists():
                dest = self.output_dir / f"{slug}.pdf"
                shutil.copy(pdf_path, dest)
                logger.info(f"📄 Paper compiled: {dest}")
                logger.info(f"📝 LaTeX source:   {dest_tex}")
                return dest
            else:
                logger.error(f"PDF compilation failed — LaTeX saved to {dest_tex}")
                return dest_tex

    def _run_compiler(self, workdir: Path, tex_file: Path) -> Path | None:
        cmd = [
            self.compiler,
            "-interaction=nonstopmode",
            "-output-directory", str(workdir),
            str(tex_file),
        ]
        for run in range(2):  # run twice for cross-references
            logger.info(f"Running {self.compiler} (pass {run+1}/2)...")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0 and run == 1:
                    logger.error(f"LaTeX error:\n{result.stdout[-2000:]}")
            except subprocess.TimeoutExpired:
                logger.error("pdflatex timed out")
                return None
            except FileNotFoundError:
                logger.error(f"{self.compiler} not found — install texlive")
                return None

        pdf = workdir / tex_file.with_suffix(".pdf").name
        return pdf if pdf.exists() else None

    @staticmethod
    def _slugify(text: str) -> str:
        text = re.sub(r"[^\w\s-]", "", text.lower())
        return re.sub(r"[-\s]+", "_", text).strip("_")[:80]
