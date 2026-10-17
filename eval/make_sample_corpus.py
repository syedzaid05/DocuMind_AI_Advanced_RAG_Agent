"""Write a small sample corpus PDF used by the evaluation harness."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

PAGE_1 = (
    "The Eiffel Tower\n\n"
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "It was completed in 1889 and stands 330 meters tall. The tower was designed "
    "by the engineering company of Gustave Eiffel. It has three levels for visitors, "
    "with restaurants on the first and second levels. The tower is repainted about "
    "every seven years to protect it from rust."
)
PAGE_2 = (
    "The Louvre Museum\n\n"
    "The Louvre is the world's largest art museum and a historic monument in Paris. "
    "It is housed in the Louvre Palace, originally built as a fortress in the late "
    "12th century. The museum opened in 1793 and is home to the Mona Lisa. Its iconic "
    "glass pyramid, designed by I. M. Pei, was completed in 1989."
)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "sample_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, PAGE_1)
    pdf.add_page()
    pdf.multi_cell(0, 8, PAGE_2)
    path = out_dir / "sample_corpus.pdf"
    pdf.output(str(path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
