"""拼接各章 → paper.html，再用 headless Chrome 打印为 PDF"""
import os, re, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
parts = ["ch1.html", "ch2_4.html", "ch4b.html", "ch5_6.html"]
body = "\n".join(open(os.path.join(H, p), encoding="utf-8").read() for p in parts)
html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>TrainScope 期末论文</title>
<link rel="stylesheet" href="style.css"></head><body>{body}</body></html>"""
open(os.path.join(H, "paper.html"), "w", encoding="utf-8").write(html)
text = re.sub(r"<pre>.*?</pre>", "", body, flags=re.S)
text = re.sub(r"<[^>]+>", "", text)
cjk = len(re.findall(r"[一-鿿]", text))
words = len(re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text))
print(f"汉字 {cjk}，英文词/数字 {words}，合计约 {cjk + words} 字")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "23307130324_杜禹衡_期末论文.pdf")
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                "--virtual-time-budget=3000", f"--print-to-pdf={out}", "file://" + os.path.join(H, "paper.html")],
               check=True, stderr=subprocess.DEVNULL)
print("PDF:", out)
