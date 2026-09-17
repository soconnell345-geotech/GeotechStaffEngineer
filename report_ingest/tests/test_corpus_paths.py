"""Finding the files: IDs here, original names on the cluster.

The corpus in this repo is a renamed copy -- ``R01.pdf`` and a manifest. The
same reports are already on the owner's cluster under the names their authors
gave them, and the Azure Document Intelligence results are there in the
uncompressed form Funhouse wrote them in. Both layouts have to load, so both
are built here out of synthetic files: a manifest of three invented reports,
empty PDFs, and DI results that are two keys long.

Nothing here opens a PDF, calls a model or needs the private corpus.
"""

from __future__ import annotations

import gzip
import json

import pytest

from report_ingest.corpus import Corpus

MANIFEST = """# A synthetic manifest, in the shape of the real one

| ID | file | origin | pages | text pages | image-only | blank | text-over-image | avg chars | sizes (in) |
|---|---|---|---|---|---|---|---|---|---|
| R01 | `Reports_PDF/Alpha Site Report.pdf` | public (test) | 85 | 85 | 0 | 0 | 0 | 2039 | 8.5x11.0x62 |
| R02 | `Beta-Report_2021.pdf` | private | 40 | 40 | 0 | 0 | 0 | 1800 | 8.5x11.0x40 |
| R03 | `Reports_PDF/Gamma Reeport (final).pdf` | private | 12 | 0 | 12 | 0 | 0 | 0 | 8.5x11.0x12 |
"""

#: What the manifest calls each report, as a test may name a file.
SOURCE = {
    "R01": "Alpha Site Report.pdf",
    "R02": "Beta-Report_2021.pdf",
    "R03": "Gamma Reeport (final).pdf",
}


def _pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    return path


@pytest.fixture()
def manifest(tmp_path):
    """The manifest on its own, away from the reports."""
    path = tmp_path / "small" / "MANIFEST.md"
    path.parent.mkdir(parents=True)
    path.write_text(MANIFEST, encoding="utf-8")
    return path


# -- the two report layouts ---------------------------------------------------

def test_the_repo_layout_is_read_by_id(tmp_path, manifest):
    reports = tmp_path / "corpus"
    for rid in SOURCE:
        _pdf(reports / f"{rid}.pdf")
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.available
    assert corpus.present_ids() == ["R01", "R02", "R03"]
    assert corpus.pdf_path("R02").name == "R02.pdf"
    assert corpus.report("R01").pages == 85


def test_the_cluster_layout_is_read_by_the_manifests_source_file(tmp_path,
                                                                manifest):
    # Flat folder of original names: R01's manifest path carries a
    # `Reports_PDF/` folder that is not there, and must be tried without it.
    reports = tmp_path / "reports"
    for name in SOURCE.values():
        _pdf(reports / name)
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.available
    assert corpus.present_ids() == ["R01", "R02", "R03"]
    assert corpus.pdf_path("R01").name == "Alpha Site Report.pdf"
    assert corpus.pdf_path("R02").name == "Beta-Report_2021.pdf"
    # The IDs are still the only thing the run says out loud.
    assert repr(corpus.report("R01")) == (
        "ReportInfo(R01, pages=85, text_pages=85, image_only=0)")


def test_the_source_folder_is_used_when_it_is_really_there(tmp_path,
                                                           manifest):
    reports = tmp_path / "reports"
    _pdf(reports / "Reports_PDF" / SOURCE["R01"])
    _pdf(reports / SOURCE["R02"])
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.pdf_path("R01").parent.name == "Reports_PDF"
    assert corpus.pdf_path("R02").parent == reports


def test_a_report_filed_under_some_other_folder_is_found_by_its_name(
        tmp_path, manifest):
    # Neither `Reports_PDF/<name>` nor `<name>` exists at the top level, so
    # the base name over a walk of the folder is what finds it.
    reports = tmp_path / "reports"
    _pdf(reports / "2019" / "scans" / SOURCE["R01"])
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.present_ids() == ["R01"]
    assert corpus.pdf_path("R01").parent.name == "scans"


def test_punctuation_and_spacing_differences_still_resolve(tmp_path,
                                                           manifest):
    reports = tmp_path / "reports"
    _pdf(reports / "gamma_reeport_final.pdf")       # R03, differently written
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.present_ids() == ["R03"]
    assert corpus.pdf_path("R03").name == "gamma_reeport_final.pdf"


def test_a_report_that_is_not_in_the_folder_is_absent_not_guessed(tmp_path,
                                                                  manifest):
    reports = tmp_path / "reports"
    _pdf(reports / SOURCE["R02"])
    corpus = Corpus(reports, manifest=manifest)

    assert corpus.present_ids() == ["R02"]
    with pytest.raises(FileNotFoundError, match="no PDF for R01"):
        corpus.pdf_path("R01")


def test_original_names_with_no_manifest_say_what_is_missing(tmp_path):
    reports = tmp_path / "reports"
    for name in SOURCE.values():
        _pdf(reports / name)
    corpus = Corpus(reports)                        # no manifest anywhere

    assert not corpus.available
    with pytest.raises(FileNotFoundError, match="original file names"):
        corpus.pdf_path("R01")


def test_id_named_pdfs_alone_still_load_without_a_manifest(tmp_path):
    reports = tmp_path / "corpus"
    _pdf(reports / "R07.pdf")
    corpus = Corpus(reports)

    assert corpus.available
    assert [r.id for r in corpus.list_reports()] == ["R07"]
    assert corpus.report("R07").pages == 0          # counted, never guessed


# -- the two DI forms ---------------------------------------------------------

def _gz(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(blob, fh)
    return path


def _json(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


@pytest.fixture()
def cluster_corpus(tmp_path, manifest):
    reports = tmp_path / "reports"
    for name in SOURCE.values():
        _pdf(reports / name)
    return reports, tmp_path / "di", manifest


def test_the_uncompressed_di_data_form_is_found_and_read(cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    _json(di_dir / "DI_data_Alpha Site Report.json", {"pages": ["a"]})
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert corpus.has_di("R01")
    assert corpus.load_di("R01") == {"pages": ["a"]}


def test_the_gzipped_id_form_is_found_and_read(cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    _gz(di_dir / "R02.json.gz", {"pages": ["b"]})
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert corpus.has_di("R02")
    assert corpus.load_di("R02") == {"pages": ["b"]}


def test_gzip_wins_when_a_report_has_both(cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    _gz(di_dir / "R01.json.gz", {"which": "gz"})
    _json(di_dir / "DI_data_Alpha Site Report.json", {"which": "raw"})
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert corpus.di_file("R01").name == "R01.json.gz"
    assert corpus.load_di("R01") == {"which": "gz"}


def test_the_di_file_is_sniffed_rather_than_trusted_from_its_suffix(
        cluster_corpus):
    # A gzipped export saved as plain .json reads anyway, and so does the
    # other way round: the first two bytes decide.
    reports, di_dir, manifest = cluster_corpus
    _gz(di_dir / "R02.json", {"pages": ["b"]})
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert corpus.load_di("R02") == {"pages": ["b"]}


def test_a_di_name_written_differently_is_still_matched(cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    _json(di_dir / "di_data_alpha_site_report.json", {"pages": ["a"]})
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert corpus.has_di("R01")


def test_no_di_result_is_not_an_error(cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    di_dir.mkdir()
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    assert not corpus.has_di("R03")
    assert corpus.load_di("R03") is None


def test_an_unreadable_di_result_warns_and_reads_as_unavailable(
        cluster_corpus):
    reports, di_dir, manifest = cluster_corpus
    di_dir.mkdir()
    (di_dir / "DI_data_Beta-Report_2021.json").write_bytes(b"\x1f\x8b broken")
    corpus = Corpus(reports, manifest=manifest, di_dir=di_dir)

    with pytest.warns(RuntimeWarning, match="unusable"):
        assert corpus.load_di("R02") is None
