"""
Export reports/report.md to reports/report.html and reports/report.pdf.
Uses mistune to parse markdown and Microsoft Edge headless to print to PDF.
"""
from pathlib import Path
import subprocess
import mistune

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
MD_PATH = REPORT_DIR / "report.md"
HTML_PATH = REPORT_DIR / "report.html"
PDF_PATH = REPORT_DIR / "report.pdf"

CSS = """
@page {
    size: A4;
    margin: 12mm 14mm 12mm 14mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1f2328;
    line-height: 1.4;
    font-size: 11px;
    max-width: 100%;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 18px;
    border-bottom: 2px solid #0969da;
    padding-bottom: 4px;
    margin-top: 0;
    margin-bottom: 8px;
    color: #1f2328;
}
h2 {
    font-size: 14px;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
    color: #0969da;
    page-break-after: avoid;
}
h3 {
    font-size: 12px;
    margin-top: 10px;
    margin-bottom: 4px;
    color: #24292f;
    page-break-after: avoid;
}
p {
    margin: 5px 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 10px;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 3px 6px;
    text-align: left;
}
th {
    background-color: #f6f8fa;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #fcfdfd;
}
blockquote {
    margin: 6px 0;
    padding: 4px 10px;
    color: #57606a;
    background-color: #f6f8fa;
    border-left: 3px solid #0969da;
    font-size: 10.5px;
}
code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 10px;
    background-color: rgba(175, 184, 193, 0.2);
    padding: 1px 3px;
    border-radius: 3px;
}
pre {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 8px;
    overflow-x: auto;
    font-size: 9.5px;
    line-height: 1.35;
    margin: 6px 0;
    page-break-inside: avoid;
}
pre code {
    background-color: transparent;
    padding: 0;
}
img {
    max-width: 65%;
    max-height: 220px;
    object-fit: contain;
    display: block;
    margin: 8px auto;
    border: 1px solid #e1e4e8;
    border-radius: 3px;
    page-break-inside: avoid;
}
em {
    color: #57606a;
    font-size: 10px;
}
hr {
    border: 0;
    height: 1px;
    background: #d0d7de;
    margin: 12px 0;
}
ol, ul {
    margin: 4px 0 6px 0;
    padding-left: 20px;
}
li {
    margin: 2px 0;
}
@media print {
    body {
        max-width: 100%;
        padding: 0;
    }
}
"""

def main():
    print(f"Reading {MD_PATH}...")
    raw_md = MD_PATH.read_text(encoding="utf-8")
    
    # Use mistune with gfm plugins
    markdown = mistune.create_markdown(plugins=['table', 'strikethrough'])
    html_content = markdown(raw_md)
    
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>P11 — Online News Popularity Report</title>
<style>{CSS}</style>
</head>
<body>
{html_content}
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Saved HTML to {HTML_PATH}")
    
    # Check for Microsoft Edge
    edge_exe = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if edge_exe.exists():
        print(f"Exporting PDF via Microsoft Edge headless...")
        cmd = [
            str(edge_exe),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            str(HTML_PATH)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and PDF_PATH.exists():
            print(f"Successfully generated {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
        else:
            print("Edge print command output:", res.stdout, res.stderr)
    else:
        print("Microsoft Edge executable not found; HTML output is ready for printing.")

if __name__ == "__main__":
    main()
