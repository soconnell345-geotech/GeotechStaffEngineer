"""Tests for diggs_parser.py — DIGGS 2.6 XML parser."""

import pytest

from subsurface_characterization.diggs_parser import parse_diggs
from subsurface_characterization.results import DiggsParseResult


class TestParseSingleBoring:
    def test_parse_basic(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        assert isinstance(result, DiggsParseResult)
        assert result.n_investigations == 1
        assert result.site.project_name == "Test Project"

    def test_coordinates(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.x == pytest.approx(100.0)
        assert inv.y == pytest.approx(200.0)
        assert inv.elevation_m == pytest.approx(10.0)

    def test_total_depth(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.total_depth_m == pytest.approx(15.0)

    def test_lithology(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert len(inv.lithology) == 3
        assert inv.lithology[0].uscs == "SM"
        assert inv.lithology[1].description == "Soft gray clay"
        assert inv.lithology[2].top_depth_m == pytest.approx(10.0)

    def test_spt_data(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        spt = inv.get_measurements("N_spt")
        assert len(spt) == 4
        assert spt[0].depth_m == pytest.approx(1.5)
        assert spt[0].value == pytest.approx(8)
        assert spt[-1].value == pytest.approx(30)

    def test_gwl(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.gwl_depth_m == pytest.approx(3.5)


class TestParseMultiBoring:
    def test_two_borings(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        assert result.n_investigations == 2
        ids = result.site.investigation_ids()
        assert "B-1" in ids
        assert "B-2" in ids

    def test_spt_association(self, multi_boring_diggs_xml):
        """SPT tests are associated with correct borings via xlink:href."""
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        assert len(b1.get_measurements("N_spt")) == 2
        assert len(b2.get_measurements("N_spt")) == 2

    def test_atterberg(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        ll = b1.get_measurements("LL_pct")
        pl = b1.get_measurements("PL_pct")
        assert len(ll) == 1
        assert ll[0].value == pytest.approx(55)
        assert len(pl) == 1
        assert pl[0].value == pytest.approx(22)

    def test_moisture(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        wn = b1.get_measurements("wn_pct")
        assert len(wn) == 1
        assert wn[0].value == pytest.approx(42)

    def test_gwl_per_boring(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        assert b1.gwl_depth_m == pytest.approx(3.5)
        assert b2.gwl_depth_m == pytest.approx(4.0)

    def test_lithology_counts(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        assert result.n_lithology_intervals == 6  # 3 + 3


class TestNamespaceDetection:
    def test_25a_namespace(self, diggs_25a_xml):
        result = parse_diggs(content=diggs_25a_xml)
        assert result.n_investigations == 1
        inv = result.site.investigations[0]
        assert inv.investigation_id == "B-1"
        assert inv.x == pytest.approx(50.0)

    def test_25a_spt(self, diggs_25a_xml):
        result = parse_diggs(content=diggs_25a_xml)
        inv = result.site.investigations[0]
        spt = inv.get_measurements("N_spt")
        assert len(spt) == 1
        assert spt[0].value == pytest.approx(12)


class TestErrorHandling:
    def test_no_input(self):
        with pytest.raises(ValueError, match="Must provide"):
            parse_diggs()

    def test_invalid_xml(self):
        with pytest.raises(ValueError, match="Invalid XML"):
            parse_diggs(content="<not valid xml")

    def test_empty_xml(self):
        """XML with no borings should return empty SiteModel."""
        xml = """<?xml version="1.0"?>
        <Diggs xmlns="http://diggsml.org/schemas/2.6"
               xmlns:gml="http://www.opengis.net/gml/3.2">
          <Project><gml:name>Empty</gml:name></Project>
        </Diggs>"""
        result = parse_diggs(content=xml)
        assert result.n_investigations == 0

    def test_summary(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        s = result.summary()
        assert "Test Project" in s
        assert "Investigations: 1" in s

    def test_to_dict(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        d = result.to_dict()
        assert d["n_investigations"] == 1
        assert d["project_name"] == "Test Project"
        assert d["site"] is not None


# ======================================================================
# Tier 1: CPT, Vane, Consolidation, Triaxial, Direct Shear, UCS
# ======================================================================

class TestParseCPT:
    def test_single_depth_elements(self, cpt_diggs_xml):
        result = parse_diggs(content=cpt_diggs_xml)
        inv = result.site.investigations[0]
        qc = inv.get_measurements("qc_kPa")
        assert len(qc) == 2
        assert qc[0].depth_m == pytest.approx(1.0)
        assert qc[0].value == pytest.approx(2500)
        assert qc[1].value == pytest.approx(4800)

    def test_cpt_sleeve_friction(self, cpt_diggs_xml):
        result = parse_diggs(content=cpt_diggs_xml)
        inv = result.site.investigations[0]
        fs = inv.get_measurements("fs_kPa")
        assert len(fs) == 2
        assert fs[0].value == pytest.approx(35)

    def test_cpt_pore_pressure(self, cpt_diggs_xml):
        result = parse_diggs(content=cpt_diggs_xml)
        inv = result.site.investigations[0]
        u2 = inv.get_measurements("u2_kPa")
        assert len(u2) == 2
        assert u2[0].value == pytest.approx(50)

    def test_cpt_friction_ratio(self, cpt_diggs_xml):
        result = parse_diggs(content=cpt_diggs_xml)
        inv = result.site.investigations[0]
        rf = inv.get_measurements("Rf_pct")
        assert len(rf) == 2
        assert rf[0].value == pytest.approx(1.4)

    def test_cpt_source_and_type(self, cpt_diggs_xml):
        result = parse_diggs(content=cpt_diggs_xml)
        inv = result.site.investigations[0]
        qc = inv.get_measurements("qc_kPa")
        assert qc[0].source == "field"
        assert qc[0].test_type == "CPTu"

    def test_nested_readings(self, cpt_nested_diggs_xml):
        result = parse_diggs(content=cpt_nested_diggs_xml)
        inv = result.site.investigations[0]
        qc = inv.get_measurements("qc_kPa")
        assert len(qc) == 3
        assert qc[0].depth_m == pytest.approx(0.5)
        assert qc[0].value == pytest.approx(1200)
        assert qc[2].value == pytest.approx(3600)

    def test_nested_fs(self, cpt_nested_diggs_xml):
        result = parse_diggs(content=cpt_nested_diggs_xml)
        inv = result.site.investigations[0]
        fs = inv.get_measurements("fs_kPa")
        assert len(fs) == 3
        assert fs[1].value == pytest.approx(30)


class TestParseVane:
    def test_peak_strength(self, vane_diggs_xml):
        result = parse_diggs(content=vane_diggs_xml)
        inv = result.site.investigations[0]
        su = inv.get_measurements("Su_vane_kPa")
        assert len(su) == 1
        assert su[0].depth_m == pytest.approx(5.0)
        assert su[0].value == pytest.approx(45)

    def test_remolded_strength(self, vane_diggs_xml):
        result = parse_diggs(content=vane_diggs_xml)
        inv = result.site.investigations[0]
        su_rem = inv.get_measurements("Su_vane_remolded_kPa")
        assert len(su_rem) == 1
        assert su_rem[0].value == pytest.approx(9)

    def test_sensitivity(self, vane_diggs_xml):
        result = parse_diggs(content=vane_diggs_xml)
        inv = result.site.investigations[0]
        st = inv.get_measurements("St")
        assert len(st) == 1
        assert st[0].value == pytest.approx(5)

    def test_computed_sensitivity(self, vane_computed_st_xml):
        """Sensitivity computed from peak/remolded when not given."""
        result = parse_diggs(content=vane_computed_st_xml)
        inv = result.site.investigations[0]
        st = inv.get_measurements("St")
        assert len(st) == 1
        assert st[0].value == pytest.approx(6.0)

    def test_vane_source(self, vane_diggs_xml):
        result = parse_diggs(content=vane_diggs_xml)
        inv = result.site.investigations[0]
        su = inv.get_measurements("Su_vane_kPa")
        assert su[0].source == "field"
        assert su[0].test_type == "vane_shear"


class TestParseConsolidation:
    def test_all_params(self, consolidation_diggs_xml):
        result = parse_diggs(content=consolidation_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("e0")[0].value == pytest.approx(0.95)
        assert inv.get_measurements("Cc")[0].value == pytest.approx(0.35)
        assert inv.get_measurements("Cr")[0].value == pytest.approx(0.06)
        assert inv.get_measurements("sigma_p_kPa")[0].value == pytest.approx(120)

    def test_depth(self, consolidation_diggs_xml):
        result = parse_diggs(content=consolidation_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("Cc")[0].depth_m == pytest.approx(6.0)

    def test_source_lab(self, consolidation_diggs_xml):
        result = parse_diggs(content=consolidation_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("Cc")[0].source == "lab"
        assert inv.get_measurements("Cc")[0].test_type == "consolidation"


class TestParseTriaxial:
    def test_uu_cu(self, triaxial_uu_diggs_xml):
        result = parse_diggs(content=triaxial_uu_diggs_xml)
        inv = result.site.investigations[0]
        cu = inv.get_measurements("cu_kPa")
        assert len(cu) == 1
        assert cu[0].value == pytest.approx(75)
        assert cu[0].depth_m == pytest.approx(5.0)

    def test_cd_phi_and_c(self, triaxial_cd_diggs_xml):
        result = parse_diggs(content=triaxial_cd_diggs_xml)
        inv = result.site.investigations[0]
        phi = inv.get_measurements("phi_deg")
        c = inv.get_measurements("c_kPa")
        assert len(phi) == 1
        assert phi[0].value == pytest.approx(32)
        assert len(c) == 1
        assert c[0].value == pytest.approx(5)

    def test_triaxial_source(self, triaxial_uu_diggs_xml):
        result = parse_diggs(content=triaxial_uu_diggs_xml)
        inv = result.site.investigations[0]
        cu = inv.get_measurements("cu_kPa")
        assert cu[0].source == "lab"
        assert cu[0].test_type == "triaxial"


class TestParseDirectShear:
    def test_phi_and_c(self, direct_shear_diggs_xml):
        result = parse_diggs(content=direct_shear_diggs_xml)
        inv = result.site.investigations[0]
        phi = inv.get_measurements("phi_deg")
        c = inv.get_measurements("c_kPa")
        assert len(phi) == 1
        assert phi[0].value == pytest.approx(28)
        assert phi[0].depth_m == pytest.approx(3.0)
        assert len(c) == 1
        assert c[0].value == pytest.approx(10)

    def test_direct_shear_source(self, direct_shear_diggs_xml):
        result = parse_diggs(content=direct_shear_diggs_xml)
        inv = result.site.investigations[0]
        phi = inv.get_measurements("phi_deg")
        assert phi[0].source == "lab"
        assert phi[0].test_type == "direct_shear"


class TestParseUCS:
    def test_qu(self, ucs_diggs_xml):
        result = parse_diggs(content=ucs_diggs_xml)
        inv = result.site.investigations[0]
        qu = inv.get_measurements("qu_kPa")
        assert len(qu) == 1
        assert qu[0].value == pytest.approx(150)
        assert qu[0].depth_m == pytest.approx(4.0)

    def test_ucs_source(self, ucs_diggs_xml):
        result = parse_diggs(content=ucs_diggs_xml)
        inv = result.site.investigations[0]
        qu = inv.get_measurements("qu_kPa")
        assert qu[0].source == "lab"
        assert qu[0].test_type == "UCS"


# ======================================================================
# Tier 2: Point Load, Specific Gravity, Gradation, Permeability,
#          Compaction, Dynamic Probe
# ======================================================================

class TestParsePointLoad:
    def test_is50(self, point_load_diggs_xml):
        result = parse_diggs(content=point_load_diggs_xml)
        inv = result.site.investigations[0]
        is50 = inv.get_measurements("Is50_MPa")
        assert len(is50) == 1
        assert is50[0].value == pytest.approx(3.5)
        assert is50[0].depth_m == pytest.approx(12.0)

    def test_point_load_source(self, point_load_diggs_xml):
        result = parse_diggs(content=point_load_diggs_xml)
        inv = result.site.investigations[0]
        is50 = inv.get_measurements("Is50_MPa")
        assert is50[0].source == "lab"
        assert is50[0].test_type == "point_load"


class TestParseSpecificGravity:
    def test_gs(self, specific_gravity_diggs_xml):
        result = parse_diggs(content=specific_gravity_diggs_xml)
        inv = result.site.investigations[0]
        gs = inv.get_measurements("Gs")
        assert len(gs) == 1
        assert gs[0].value == pytest.approx(2.68)
        assert gs[0].depth_m == pytest.approx(5.0)

    def test_gs_source(self, specific_gravity_diggs_xml):
        result = parse_diggs(content=specific_gravity_diggs_xml)
        inv = result.site.investigations[0]
        gs = inv.get_measurements("Gs")
        assert gs[0].source == "lab"
        assert gs[0].test_type == "specific_gravity"


class TestParseGradation:
    def test_percentages(self, gradation_diggs_xml):
        result = parse_diggs(content=gradation_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("pct_gravel")[0].value == pytest.approx(15)
        assert inv.get_measurements("pct_sand")[0].value == pytest.approx(55)
        assert inv.get_measurements("pct_fines")[0].value == pytest.approx(30)

    def test_grain_sizes(self, gradation_diggs_xml):
        result = parse_diggs(content=gradation_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("D10_mm")[0].value == pytest.approx(0.002)
        assert inv.get_measurements("D30_mm")[0].value == pytest.approx(0.05)
        assert inv.get_measurements("D60_mm")[0].value == pytest.approx(0.5)

    def test_gradation_source(self, gradation_diggs_xml):
        result = parse_diggs(content=gradation_diggs_xml)
        inv = result.site.investigations[0]
        m = inv.get_measurements("pct_sand")[0]
        assert m.source == "lab"
        assert m.test_type == "gradation"

    def test_alternate_tag(self, gradation_alt_tag_xml):
        """MaterialGradationTest also parsed."""
        result = parse_diggs(content=gradation_alt_tag_xml)
        inv = result.site.investigations[0]
        assert inv.get_measurements("pct_gravel")[0].value == pytest.approx(40)
        assert inv.get_measurements("pct_sand")[0].value == pytest.approx(45)
        assert inv.get_measurements("pct_fines")[0].value == pytest.approx(15)


class TestParsePermeability:
    def test_k(self, permeability_diggs_xml):
        result = parse_diggs(content=permeability_diggs_xml)
        inv = result.site.investigations[0]
        k = inv.get_measurements("k_m_per_s")
        assert len(k) == 1
        assert k[0].value == pytest.approx(1.5e-8)
        assert k[0].depth_m == pytest.approx(7.0)

    def test_permeability_source(self, permeability_diggs_xml):
        result = parse_diggs(content=permeability_diggs_xml)
        inv = result.site.investigations[0]
        k = inv.get_measurements("k_m_per_s")
        assert k[0].source == "lab"
        assert k[0].test_type == "permeability"


class TestParseCompaction:
    def test_max_density_and_omc(self, compaction_diggs_xml):
        result = parse_diggs(content=compaction_diggs_xml)
        inv = result.site.investigations[0]
        gamma = inv.get_measurements("gamma_max_kNm3")
        wn_opt = inv.get_measurements("wn_opt_pct")
        assert len(gamma) == 1
        assert gamma[0].value == pytest.approx(19.5)
        assert len(wn_opt) == 1
        assert wn_opt[0].value == pytest.approx(12.5)

    def test_compaction_source(self, compaction_diggs_xml):
        result = parse_diggs(content=compaction_diggs_xml)
        inv = result.site.investigations[0]
        gamma = inv.get_measurements("gamma_max_kNm3")
        assert gamma[0].source == "lab"
        assert gamma[0].test_type == "compaction"


class TestParseDynamicProbe:
    def test_blow_counts(self, dynamic_probe_diggs_xml):
        result = parse_diggs(content=dynamic_probe_diggs_xml)
        inv = result.site.investigations[0]
        ndp = inv.get_measurements("N_dp")
        assert len(ndp) == 2
        assert ndp[0].depth_m == pytest.approx(3.0)
        assert ndp[0].value == pytest.approx(18)
        assert ndp[1].value == pytest.approx(25)

    def test_dynamic_probe_source(self, dynamic_probe_diggs_xml):
        result = parse_diggs(content=dynamic_probe_diggs_xml)
        inv = result.site.investigations[0]
        ndp = inv.get_measurements("N_dp")
        assert ndp[0].source == "field"
        assert ndp[0].test_type == "dynamic_probe"


# ======================================================================
# Tier 3: Pressuremeter, DMT
# ======================================================================

class TestParsePressuremeter:
    def test_modulus_and_limit(self, pressuremeter_diggs_xml):
        result = parse_diggs(content=pressuremeter_diggs_xml)
        inv = result.site.investigations[0]
        e_pmt = inv.get_measurements("E_pmt_kPa")
        p_limit = inv.get_measurements("p_limit_kPa")
        assert len(e_pmt) == 1
        assert e_pmt[0].value == pytest.approx(15000)
        assert e_pmt[0].depth_m == pytest.approx(8.0)
        assert len(p_limit) == 1
        assert p_limit[0].value == pytest.approx(1200)

    def test_pressuremeter_source(self, pressuremeter_diggs_xml):
        result = parse_diggs(content=pressuremeter_diggs_xml)
        inv = result.site.investigations[0]
        e_pmt = inv.get_measurements("E_pmt_kPa")
        assert e_pmt[0].source == "field"
        assert e_pmt[0].test_type == "pressuremeter"


class TestParseDMT:
    def test_dmt_params(self, dmt_diggs_xml):
        result = parse_diggs(content=dmt_diggs_xml)
        inv = result.site.investigations[0]
        ed = inv.get_measurements("ED_kPa")
        id_dmt = inv.get_measurements("ID_dmt")
        kd = inv.get_measurements("KD_dmt")
        assert len(ed) == 1
        assert ed[0].value == pytest.approx(22000)
        assert ed[0].depth_m == pytest.approx(5.0)
        assert len(id_dmt) == 1
        assert id_dmt[0].value == pytest.approx(1.8)
        assert len(kd) == 1
        assert kd[0].value == pytest.approx(4.5)

    def test_dmt_source(self, dmt_diggs_xml):
        result = parse_diggs(content=dmt_diggs_xml)
        inv = result.site.investigations[0]
        ed = inv.get_measurements("ED_kPa")
        assert ed[0].source == "field"
        assert ed[0].test_type == "DMT"


# ======================================================================
# Cross-cutting: xlink association with multiple borings
# ======================================================================

class TestMultiBoringXlink:
    def test_vane_to_correct_boring(self, multi_boring_xlink_xml):
        result = parse_diggs(content=multi_boring_xlink_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        # Vane: BH1 at 5m, BH2 at 6m
        su_b1 = b1.get_measurements("Su_vane_kPa")
        su_b2 = b2.get_measurements("Su_vane_kPa")
        assert len(su_b1) == 1
        assert su_b1[0].value == pytest.approx(40)
        assert len(su_b2) == 1
        assert su_b2[0].value == pytest.approx(55)

    def test_consolidation_to_correct_boring(self, multi_boring_xlink_xml):
        result = parse_diggs(content=multi_boring_xlink_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        # Consolidation only in BH2
        assert len(b1.get_measurements("Cc")) == 0
        cc_b2 = b2.get_measurements("Cc")
        assert len(cc_b2) == 1
        assert cc_b2[0].value == pytest.approx(0.40)

    def test_cpt_to_correct_boring(self, multi_boring_xlink_xml):
        result = parse_diggs(content=multi_boring_xlink_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        # CPT only in BH1
        qc_b1 = b1.get_measurements("qc_kPa")
        assert len(qc_b1) == 1
        assert qc_b1[0].value == pytest.approx(5000)
        assert len(b2.get_measurements("qc_kPa")) == 0

    def test_total_measurement_count(self, multi_boring_xlink_xml):
        result = parse_diggs(content=multi_boring_xlink_xml)
        # BH1: Su_vane(1) + qc(1) + fs(1) = 3
        # BH2: Su_vane(1) + Cc(1) + sigma_p(1) = 3
        assert result.n_measurements == 6
