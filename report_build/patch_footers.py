"""docx 后处理：页脚 PAGE 域加格式开关 + 删除封面节空 pgNumType（WPS 兼容）。

第 2 节（前置）页脚应为 ROMAN，第 3 节（正文）应为 arabic。
docx-js 的 PageNumber.CURRENT 只生成裸 PAGE 域，需要按节补 \\* ROMAN / \\* arabic。
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
tmp = path.with_suffix(".tmp.docx")
shutil.copy(path, tmp)

with zipfile.ZipFile(tmp) as z:
    names = z.namelist()
    files = {n: z.read(n) for n in names}

# 1) document.xml：定位各节引用的 footer rId，并删除封面节空 pgNumType
doc = files["word/document.xml"].decode("utf-8")
doc = doc.replace("<w:pgNumType/>", "")
files["word/document.xml"] = doc.encode("utf-8")

# sectPr 顺序 = 节顺序（节 1 封面无 footers，节 2 罗马，节 3 阿拉伯）
sect_prs = re.findall(r"<w:sectPr.*?</w:sectPr>", doc, re.S)
print(f"sections: {len(sect_prs)}")
rels = files["word/_rels/document.xml.rels"].decode("utf-8")
rid_to_file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))

footer_fmt = {}  # footer file -> ROMAN / arabic
for i, sp in enumerate(sect_prs):
    m = re.search(r'w:footerReference[^>]*r:id="(rId\d+)"', sp)
    if not m:
        continue
    f = rid_to_file.get(m.group(1))
    if not f:
        continue
    if 'w:fmt="upperRoman"' in sp:
        footer_fmt[f] = "ROMAN"
    else:
        footer_fmt[f] = "arabic"
print("footer fmt map:", footer_fmt)

for f, fmt in footer_fmt.items():
    key = f"word/{f}"
    if key not in files:
        continue
    xml = files[key].decode("utf-8")
    xml2 = xml.replace(
        '>PAGE<', f'>PAGE \\* {fmt} \\* MERGEFORMAT<'
    ).replace(
        '>PAGE  <', f'>PAGE \\* {fmt} \\* MERGEFORMAT  <'
    )
    # instrText 形式: <w:instrText ...>PAGE</w:instrText>
    xml2 = re.sub(
        r'(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)',
        lambda m: m.group(1) + f"PAGE \\* {fmt} \\* MERGEFORMAT" + m.group(2),
        xml2,
    )
    if xml2 != xml:
        files[key] = xml2.encode("utf-8")
        print(f"patched {f} -> {fmt}")
    else:
        print(f"WARNING: no PAGE field found in {f}")

with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
    for n, data in files.items():
        z.writestr(n, data)
tmp.unlink()
print("PATCH_DONE")
