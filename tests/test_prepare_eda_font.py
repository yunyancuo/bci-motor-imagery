"""预处理、EDA、中文字体配置及报告后处理工具单元测试。"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.io import savemat

from src.cn_font import configure_chinese_font
from src.eda import compute_erd_percentage
from src.prepare import load_mat_labels, parse_annotation_id_map, safe_extract_zip


def test_configure_chinese_font_fallbacks(tmp_path: Path) -> None:
    """验证字体配置函数在缺失或损坏字体文件时优雅回退且关闭负号乱码。"""
    missing = tmp_path / "not_exist.ttf"
    fonts = configure_chinese_font(missing)
    assert "DejaVu Sans" in fonts
    assert matplotlib.rcParams["axes.unicode_minus"] is False

    corrupted = tmp_path / "broken.ttf"
    corrupted.write_bytes(b"not a valid ttf file")
    fonts_corrupted = configure_chinese_font(corrupted)
    assert "SimHei" in fonts_corrupted
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_safe_extract_zip_blocks_zip_slip(tmp_path: Path) -> None:
    """验证 safe_extract_zip 允许正常解压并拦截 Zip Slip 路径穿越攻击。"""
    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("sub/hello.txt", "ok")
    out_good = tmp_path / "out_good"
    out_good.mkdir()
    safe_extract_zip(good_zip, out_good)
    assert (out_good / "sub" / "hello.txt").read_text(encoding="utf-8") == "ok"

    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../escaped.txt", "pwned")
    out_evil = tmp_path / "out_evil"
    out_evil.mkdir()
    with pytest.raises(ValueError, match="Zip Slip"):
        safe_extract_zip(evil_zip, out_evil)


def test_load_mat_labels_and_annotation_parser(tmp_path: Path) -> None:
    """验证 .mat 标签 1–4 到 0–3 转换、越界拦截及非数字 GDF 注释过滤。"""
    mat_ok = tmp_path / "A01E.mat"
    savemat(str(mat_ok), {"classlabel": np.array([[1], [2], [3], [4]])})
    labels = load_mat_labels(mat_ok)
    assert labels.dtype == np.int64
    assert np.array_equal(labels, np.array([0, 1, 2, 3]))

    mat_bad_key = tmp_path / "bad_key.mat"
    savemat(str(mat_bad_key), {"other": np.array([1, 2])})
    with pytest.raises(ValueError, match="classlabel"):
        load_mat_labels(mat_bad_key)

    mat_out_of_range = tmp_path / "bad_val.mat"
    savemat(str(mat_out_of_range), {"classlabel": np.array([1, 5])})
    with pytest.raises(ValueError, match="越界"):
        load_mat_labels(mat_out_of_range)

    parsed = parse_annotation_id_map({"769": 1, "770": 2, "BAD_artifact": 3, " 783 ": 4})
    assert parsed == {1: 769, 2: 770, 4: 783}


def test_compute_erd_percentage_accuracy_and_zero_baseline() -> None:
    """验证 ERD/ERS 百分比计算正确性与零基线防除零保护。"""
    times = np.linspace(-1.0, 2.0, 301)
    trials = np.ones((10, len(times)), dtype=np.float64) * 2.0  # 基线能量 = 4.0
    # 提示后幅度降为 sqrt(2)，能量降为 2.0 => 相对基线变化 -50%
    trials[:, times >= 0.0] = np.sqrt(2.0)

    erd = compute_erd_percentage(trials, times, baseline=(-1.0, 0.0))
    assert erd[times < 0.0].mean() == pytest.approx(0.0, abs=1e-6)
    assert erd[times >= 0.0].mean() == pytest.approx(-50.0, abs=1e-6)

    # 全零基线不得出现 NaN / Inf
    zero_trials = np.zeros_like(trials)
    erd_zero = compute_erd_percentage(zero_trials, times, baseline=(-1.0, 0.0))
    assert np.all(np.isfinite(erd_zero))
    assert np.all(erd_zero == 0.0)

    with pytest.raises(ValueError, match="无有效采样点"):
        compute_erd_percentage(trials, times, baseline=(-5.0, -2.0))


def test_patch_docx_footers_safe_rewrite(tmp_path: Path) -> None:
    """验证 report_build/patch_footers.py 正确修补 PAGE 域并清理临时文件。"""
    script_path = Path(__file__).resolve().parent.parent / "report_build" / "patch_footers.py"
    spec = importlib.util.spec_from_file_location("patch_footers", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    docx_path = tmp_path / "sample.docx"
    doc_xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>"
        '<w:sectPr><w:pgNumType/></w:sectPr>'
        '<w:sectPr><w:pgNumType w:fmt="upperRoman"/>'
        '<w:footerReference r:id="rId1"/></w:sectPr>'
        '<w:sectPr><w:footerReference r:id="rId2"/></w:sectPr>'
        "</w:body></w:document>"
    )
    rels_xml = (
        '<Relationships>'
        '<Relationship Id="rId1" Target="footer1.xml"/>'
        '<Relationship Id="rId2" Target="footer2.xml"/>'
        "</Relationships>"
    )
    footer_xml = '<w:ftr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:ftr>'

    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)
        zf.writestr("word/footer1.xml", footer_xml)
        zf.writestr("word/footer2.xml", footer_xml)

    mapping = mod.patch_docx_footers(docx_path)
    assert mapping == {"footer1.xml": "ROMAN", "footer2.xml": "arabic"}
    assert not docx_path.with_suffix(".tmp.docx").exists()

    with zipfile.ZipFile(docx_path) as zf:
        assert "<w:pgNumType/>" not in zf.read("word/document.xml").decode("utf-8")
        assert "PAGE \\* ROMAN \\* MERGEFORMAT" in zf.read("word/footer1.xml").decode("utf-8")
        assert "PAGE \\* arabic \\* MERGEFORMAT" in zf.read("word/footer2.xml").decode("utf-8")
