"""DOCX 后处理脚本：为页脚 PAGE 域补齐格式开关并清理空 pgNumType（兼容 WPS）。

第 2 节（前置）页脚应为 ROMAN，第 3 节（正文）应为 arabic。
docx-js 的 PageNumber.CURRENT 仅生成裸 PAGE 域，需按节补写 ``\\* ROMAN`` / ``\\* arabic``。
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


def patch_docx_footers(docx_path: Path | str) -> dict[str, str]:
    """修补指定 DOCX 文件中各节页脚的 PAGE 域格式开关。

    参数:
        docx_path: 待修补的 ``.docx`` 文件路径。

    返回:
        页脚 XML 文件名到格式开关（``"ROMAN"`` 或 ``"arabic"``）的映射字典。
    """
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到待处理的 DOCX 文件: {path}")

    tmp = path.with_suffix(".tmp.docx")
    shutil.copy(path, tmp)
    try:
        with zipfile.ZipFile(tmp) as zf:
            names = zf.namelist()
            files = {n: zf.read(n) for n in names}

        # 1) document.xml：定位各节引用的 footer rId，并移除封面节空 pgNumType
        doc = files["word/document.xml"].decode("utf-8")
        doc = doc.replace("<w:pgNumType/>", "")
        files["word/document.xml"] = doc.encode("utf-8")

        sect_prs = re.findall(r"<w:sectPr.*?</w:sectPr>", doc, re.S)
        print(f"sections: {len(sect_prs)}")
        rels = files["word/_rels/document.xml.rels"].decode("utf-8")
        rid_to_file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))

        footer_fmt: dict[str, str] = {}
        for sp in sect_prs:
            match = re.search(r'w:footerReference[^>]*r:id="(rId\d+)"', sp)
            if not match:
                continue
            footer_file = rid_to_file.get(match.group(1))
            if not footer_file:
                continue
            footer_fmt[footer_file] = "ROMAN" if 'w:fmt="upperRoman"' in sp else "arabic"
        print("footer fmt map:", footer_fmt)

        for footer_file, fmt in footer_fmt.items():
            key = f"word/{footer_file}"
            if key not in files:
                continue
            xml = files[key].decode("utf-8")
            xml2 = xml.replace(">PAGE<", f">PAGE \\* {fmt} \\* MERGEFORMAT<").replace(
                ">PAGE  <", f">PAGE \\* {fmt} \\* MERGEFORMAT  <"
            )
            xml2 = re.sub(
                r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
                lambda m, fmt_val=fmt: (
                    m.group(1) + f"PAGE \\* {fmt_val} \\* MERGEFORMAT" + m.group(2)
                ),
                xml2,
            )
            if xml2 != xml:
                files[key] = xml2.encode("utf-8")
                print(f"patched {footer_file} -> {fmt}")
            else:
                print(f"WARNING: no PAGE field found in {footer_file}")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(name, data)
    finally:
        if tmp.exists():
            tmp.unlink()

    print("PATCH_DONE")
    return footer_fmt


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser(description="修补 DOCX 页脚 PAGE 域格式开关")
    ap.add_argument("docx_path", help="待修补的 DOCX 文件路径")
    args = ap.parse_args()
    patch_docx_footers(args.docx_path)


if __name__ == "__main__":
    main()
