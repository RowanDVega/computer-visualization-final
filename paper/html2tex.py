"""把论文各章 HTML 转换为 LaTeX 正文（paper/latex/body.tex），供 main.tex \\input"""
import html, os, re

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "latex")

SYM = {  # 字体覆盖不到的符号 → 数学/宏
    "✓": r"\checkmark{}", "✕": r"$\times$", "▲": r"$\blacktriangle$", "⚠": r"$\triangle$",
    "α": r"$\alpha$", "β": r"$\beta$", "ρ": r"$\rho$", "Δ": r"$\Delta$", "‖": r"$\|$", "ℓ": r"$\ell$",
    "≈": r"$\approx$", "≤": r"$\le$", "≥": r"$\ge$", "−": r"$-$", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "×": r"$\times$", "␣": r"\textvisiblespace{}", "…": r"\ldots{}",
}


def esc(t):
    t = html.unescape(t)
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    t = t.replace(r"\textbackslash\{\}", r"\textbackslash{}").replace(r"\textasciitilde\{\}", r"\textasciitilde{}") \
         .replace(r"\textasciicircum\{\}", r"\textasciicircum{}")
    for a, b in SYM.items():
        t = t.replace(a, b)
    return t


def inline(s):
    """行内 HTML → LaTeX"""
    out, pos = [], 0
    tok = re.compile(r"<(/?)(\w+)([^>]*)>")
    stack = []
    for m in tok.finditer(s):
        if m.start() < pos:  # 已被 sup/sub/a 整体消费的内层标签
            continue
        out.append(esc(s[pos:m.start()])); pos = m.end()
        close, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if tag == "br":
            out.append(r"\\ "); continue
        if tag == "sup" and not close:
            j = s.index("</sup>", pos); inner = s[pos:j]
            mm = re.fullmatch(r"\[([\d,\s]+)\]", inner.strip())
            if mm:
                out.append(r"\cite{" + ",".join("r" + x.strip() for x in mm.group(1).split(",")) + "}")
            else:
                out.append(r"\textsuperscript{" + inline(inner) + "}")
            pos = j + len("</sup>"); continue
        if tag == "sub" and not close:
            j = s.index("</sub>", pos); out.append(r"\textsubscript{" + inline(s[pos:j]) + "}")
            pos = j + len("</sub>"); continue
        if tag == "a" and not close:
            j = s.index("</a>", pos); href = re.search(r'href="([^"]+)"', attrs).group(1)
            out.append(r"\url{" + href + "}"); pos = j + len("</a>"); continue
        cmd = {"b": r"\textbf{", "strong": r"\textbf{", "i": r"\textit{", "em": r"\textit{", "code": r"\texttt{"}.get(tag)
        if cmd:
            out.append("}" if close else cmd)
    out.append(esc(s[pos:]))
    r = "".join(out)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", r).strip()


def table(tab_html, cap):
    rows = re.findall(r"<tr>(.*?)</tr>", tab_html, re.S)
    head = re.search(r"<thead>(.*?)</thead>", tab_html, re.S)
    cells_of = lambda r: re.findall(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", r, re.S)
    hcells = cells_of(rows[0])
    spec = "".join("L" if 'class="l"' in a else "c" for _, a, _ in hcells)
    # 文本较长的左对齐列用 X 自动换行，其余保持紧凑
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + cap + "}", r"\small",
             r"\begin{tabularx}{\linewidth}{" + spec.replace("L", "X") + "}", r"\toprule"]
    for k, r in enumerate(rows):
        cs = []
        for t, a, c in cells_of(r):
            span = re.search(r'colspan="(\d+)"', a)
            txt = inline(c)
            if t == "th": txt = r"\textbf{" + txt + "}"
            cs.append(r"\multicolumn{%s}{c}{%s}" % (span.group(1), txt) if span else txt)
        lines.append(" & ".join(cs) + r" \\")
        if k == 0 and head: lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def convert(src):
    s = src
    s = re.sub(r"<h1.*?</h1>|<div class=\"subtitle\">.*?</div>|<div class=\"author\">.*?</div>", "", s, flags=re.S)
    blocks, pos = [], 0
    pat = re.compile(r"<(h2|h3|h4|pre|p|ul|ol|figure|div|table)\b([^>]*)>(.*?)</\1>", re.S)
    pending_cap = None
    for m in pat.finditer(s):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if tag == "h2":
            t = inline(inner)
            if t.startswith("参考文献"): blocks.append("%%REFS%%"); continue
            if t.startswith("附录"):
                blocks.append(r"\section*{" + t + "}" + "\n" + r"\addcontentsline{toc}{section}{" + t + "}")
                continue
            t = re.sub(r"^第.章\s*", "", t); blocks.append(r"\section{" + t + "}")
        elif tag == "h3":
            blocks.append(r"\subsection{" + re.sub(r"^\d+\.\d+\s*", "", inline(inner)) + "}")
        elif tag == "h4":
            blocks.append(r"\subsubsection*{" + inline(inner) + "}")
        elif tag == "p":
            if 'class="kw"' in attrs or "摘要：" in inner:
                continue
            if 'class="note"' in attrs: blocks.append(r"\noindent " + inline(inner))
            else: blocks.append(inline(inner))
        elif tag in ("ul", "ol"):
            env = "itemize" if tag == "ul" else "enumerate"
            items = re.findall(r"<li>(.*?)</li>", inner, re.S)
            blocks.append(r"\begin{%s}" % env + "\n" + "\n".join(r"\item " + inline(i) for i in items) + "\n" + r"\end{%s}" % env)
        elif tag == "figure":
            img = re.search(r'src="figures/([^"]+)"', inner).group(1).replace(".svg", ".png")
            cap = re.search(r"<figcaption>(.*?)</figcaption>", inner, re.S).group(1)
            cap = re.sub(r"^<b>图\d+</b>\s*", "", cap.strip())
            blocks.append("\n".join([r"\begin{figure}[htbp]", r"\centering",
                                     r"\includegraphics[width=\linewidth]{figures/" + img + "}",
                                     r"\caption{" + inline(cap) + "}", r"\end{figure}"]))
        elif tag == "pre":
            code = html.unescape(re.sub(r"<[^>]+>", "", inner))
            blocks.append(r"\begin{Verbatim}[fontsize=\small,frame=single,breaklines,breakanywhere]" + "\n" + code.rstrip() + "\n" + r"\end{Verbatim}")
        elif tag == "div":
            if 'class="tcap"' in attrs:
                pending_cap = inline(re.sub(r"^<b>表\d+</b>\s*", "", inner.strip()))
            elif 'class="refs"' in attrs:
                refs = re.findall(r"<p>\[(\d+)\]\s*(.*?)</p>", inner, re.S)
                blocks.append("%%REFLIST%%" + "\n".join(r"\bibitem{r%s} %s" % (n, inline(t)) for n, t in refs))
            elif 'class="abstract"' in attrs:
                pass
        elif tag == "table":
            blocks.append(table(inner, pending_cap or "")); pending_cap = None
    body = "\n\n".join(blocks)
    reflist = re.search(r"%%REFLIST%%(.*?)(?=\n\n|$)", body, re.S)
    if reflist:
        body = body.replace("%%REFLIST%%" + reflist.group(1), "")
        body = body.replace("%%REFS%%", r"\begin{thebibliography}{99}" + "\n" + reflist.group(1) + "\n" + r"\end{thebibliography}")
    return body


def main():
    src = "\n".join(open(os.path.join(H, p), encoding="utf-8").read() for p in ["ch1.html", "ch2_4.html", "ch4b.html", "ch5_6.html"])
    body = convert(src)
    ab = re.search(r"摘要：</b>(.*?)</p>", src, re.S).group(1)
    kw = re.search(r"关键词：</b>(.*?)</p>", src, re.S).group(1)
    # 第一章“题目与摘要”：在该节下排版摘要与关键词
    absblock = ("\\section{题目与摘要}\n\\begin{abstractbox}\n\\noindent\\textbf{\\heiti 摘\\quad 要：}" + inline(ab) +
                "\n\n\\vspace{0.5em}\\noindent\\textbf{\\heiti 关键词：}" + inline(kw) + "\n\\end{abstractbox}")
    body = body.replace("\\section{题目与摘要}", absblock, 1)
    # 公式改写为数学模式
    body = re.sub(r"\\item \\textbf\{损失残差\}.*?，刻画", lambda m: r"\item \textbf{损失残差} $r_t=\ell_t-\operatorname{median}(\ell_{t-25:t+25})$，刻画", body, count=1)
    body = re.sub(r"\\item \\textbf\{稳健 z 分数\}：.*?，其中", lambda m: r"\item \textbf{稳健 z 分数}：$z_t=(x_t-\mathrm{med})/(1.4826\cdot\mathrm{MAD})$，其中", body, count=1)
    open(os.path.join(OUT, "body.tex"), "w", encoding="utf-8").write(body + "\n")
    left = re.findall(r"<[a-z/][^>]*>", body)
    print("body.tex", len(body), "chars; leftover tags:", left[:5])


if __name__ == "__main__":
    main()
