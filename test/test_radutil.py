import unittest
from frads import util
from frads import radutil


class TestRadutil(unittest.TestCase):

    pane_property_1 = util.PaneProperty(
            "VE1-2M_Low-E", # name
            0.01, # thickness
            'coated', # gtype,
            'front', # coated_side,
            [], # wavelengths,
            [], # transmittance,
            [], # refl_front,
            [], # refl_back,
    )

    pane_property_2 = util.PaneProperty(
            "PVB_laminated", # name
            0.01, # thickness
            "laminate", # gtype,
            "front", # coated_side,
            [], # wavelengths,
            [], # transmittance,
            [], # refl_front,
            [], # refl_back,
    )

    pane_rgb_1 = util.PaneRGB(
            pane_property_1,
            (0.042, 0.049, 0.043),
            (0.065, 0.058, 0.067),
            (0.756, 0.808, 0.744),
            )

    pane_rgb_2 = util.PaneRGB(
            pane_property_2,
            (0.11, 0.11, 0.11),
            (0.11, 0.11, 0.11),
            (0.63, 0.63, 0.63),
            )

    def test_parse_primitive(self):
        pass


    def test_unpacl_primitive(self):
        pass


    def test_parse_polygon(self):
        pass


    def test_polygon2prim(self):
        pass


    def test_primitive_normal(self):
        pass


    def test_samp_dir(self):
        pass


    def test_up_vector(self):
        pass


    def test_neutral_plastic_prim(self):
        pass


    def test_neutral_trans_prim(self):
        pass


    def test_color_plastic_prim(self):
        pass


    def test_glass_prim(self):
        pass


    def test_bsdf_prim(self):
        pass


    def test_lambda_calc(self):
        pass


    def test_angle_basis_coeff(self):
        pass


    def test_opt2str(self):
        pass


    def test_reinsrc(self):
        pass


    def test_pt_inclusion(self):
        pass


    def test_gen_grid(self):
        pass

    def test_gen_blinds(self):
        pass


    def test_analyze_vert_polygon(self):
        pass


    def test_varays(self):
        pass


    def test_gen_glazing_primitive(self):
        pass


    def test_get_glazing_primitive(self):
        prim = radutil.get_glazing_primitive(
                [self.pane_rgb_1, self.pane_rgb_2])
        res = str(prim).strip().replace('\n', ' ')
        answer = "void BRTDfunc VE1-2M_Low-E+PVB_laminated 10 if(Rdot,cr(fr("
        answer += "0.11),ft(0.63),fr(0.065)),cr(fr(0.042),ft(0.756),fr(0.11)))"
        answer += " if(Rdot,cr(fr(0.11),ft(0.63),fr(0.058)),cr(fr(0.049),ft("
        answer += "0.808),fr(0.11))) if(Rdot,cr(fr(0.11),ft(0.63),fr(0.067)),"
        answer += "cr(fr(0.043),ft(0.744),fr(0.11))) ft(0.63)*ft(0.756) ft("
        answer += "0.63)*ft(0.808) ft(0.63)*ft(0.744) 0 0 0 glaze2.cal 0 9 0 "
        answer += "0 0 0 0 0 0 0 0"
        self.assertEqual(res, answer)

if __name__ == "__main__":
    unittest.main()
