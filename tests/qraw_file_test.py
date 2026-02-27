from pathlib import Path
from unittest import TestCase

from viewer.qraw_file import QRawFile, VariableType, AbscissaScale, AC, DC, TRANSIENT

FIXTURES_DIR = Path(__file__).parent / "PyQSPICE"


class TestQRawFile(TestCase):

    def test_loading_10_DC(self):
        # arrange
        filename = FIXTURES_DIR / "UJ3N065080.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "DC Transfer Characteristic")
        self.assertFalse(qraw.complex)
        self.assertTrue(qraw.stepped)
        self.assertEqual(len(qraw.variables), 12)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "V2")
        self.assertEqual(qraw.variables[0].type, VariableType.VOLTAGE)
        self.assertEqual(qraw.variables[6].index, 6)
        self.assertEqual(qraw.variables[6].name, "Id(J1)")
        self.assertEqual(qraw.variables[6].type, VariableType.CURRENT)
        self.assertEqual(qraw.variables[11].index, 11)
        self.assertEqual(qraw.variables[11].name, "P(V1)")
        self.assertEqual(qraw.variables[11].type, VariableType.POWER)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 0.0)
        self.assertEqual(qraw.abscissa_max, 1.0e+01)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables to plot
        self.assertEqual(len(qraw.plot_suggestions), 1)
        self.assertIs(qraw.plot_suggestions[0].chart_type, DC)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["Id(J1)"])

    def test_loading_20_AC(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertTrue(qraw.stepped)
        self.assertEqual(len(qraw.variables), 32)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "Frequency")
        self.assertEqual(qraw.variables[0].type, VariableType.FREQUENCY)
        self.assertEqual(qraw.variables[31].index, 31)
        self.assertEqual(qraw.variables[31].name, "X")
        self.assertEqual(qraw.variables[31].type, VariableType.PARAMETER)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 1.0e+00)
        self.assertEqual(qraw.abscissa_max, 1.0e+08)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables to plot (AC keyword maps to AC chart type)
        self.assertEqual(len(qraw.plot_suggestions), 1)
        self.assertIs(qraw.plot_suggestions[0].chart_type, AC)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["V(vout)"])

    def test_loading_22_NyquistDia(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_Nyquist.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertFalse(qraw.stepped)
        self.assertEqual(len(qraw.variables), 24)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "Frequency")
        self.assertEqual(qraw.variables[0].type, VariableType.FREQUENCY)
        self.assertEqual(qraw.variables[1].index, 1)
        self.assertEqual(qraw.variables[1].name, "V(vout)")
        self.assertEqual(qraw.variables[1].type, VariableType.VOLTAGE)
        self.assertEqual(qraw.variables[23].index, 23)
        self.assertEqual(qraw.variables[23].name, "I(COUT)")
        self.assertEqual(qraw.variables[23].type, VariableType.CURRENT)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 1.0e+00)
        self.assertEqual(qraw.abscissa_max, 1.0e+08)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables (expression V(VOUT)/V(VO) is skipped — group has no resolved variables)
        self.assertEqual(len(qraw.plot_suggestions), 1)
        self.assertIs(qraw.plot_suggestions[0].chart_type, AC)
        self.assertEqual(qraw.plot_suggestions[0].variables, [])

    def test_loading_30_TRAN(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertFalse(qraw.stepped)
        self.assertEqual(len(qraw.variables), 47)
        self.assertEqual(qraw.num_points, 160387)
        self.assertEqual(len(qraw.variables), 47)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "Time")
        self.assertEqual(qraw.variables[0].type, VariableType.TIME)
        self.assertEqual(qraw.variables[1].index, 1)
        self.assertEqual(qraw.variables[1].name, "V(out)")
        self.assertEqual(qraw.variables[1].type, VariableType.VOLTAGE)
        self.assertEqual(qraw.variables[46].index, 46)
        self.assertEqual(qraw.variables[46].name, "I(C1)")
        self.assertEqual(qraw.variables[46].type, VariableType.CURRENT)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 0.0)
        self.assertEqual(qraw.abscissa_max, 1.0e-03)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables — three «...» groups each become their own PlotSuggestion
        self.assertEqual(len(qraw.plot_suggestions), 3)
        self.assertIs(qraw.plot_suggestions[0].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["V(out)", "V(ss)", "V(in)"])
        self.assertIs(qraw.plot_suggestions[1].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[1].variables], ["V(ics)", "V(comp)"])
        self.assertIs(qraw.plot_suggestions[2].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[2].variables], ["V(on)", "V(off)"])

    def test_loading_40_Bode(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_Bode.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertFalse(qraw.stepped)
        self.assertEqual(len(qraw.variables), 2)
        self.assertEqual(qraw.num_points, 4488)
        self.assertEqual(len(qraw.variables), 2)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "Frequency")
        self.assertEqual(qraw.variables[0].type, VariableType.FREQUENCY)
        self.assertEqual(qraw.variables[1].index, 1)
        self.assertEqual(qraw.variables[1].name, "OpenLoopGain")
        self.assertEqual(qraw.variables[1].type, VariableType.VOLTAGE)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 1.0e+03)
        self.assertEqual(qraw.abscissa_max, 1.0e+05)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables to plot (no mode keyword — inferred from FREQUENCY x-axis → AC)
        self.assertEqual(len(qraw.plot_suggestions), 1)
        self.assertIs(qraw.plot_suggestions[0].chart_type, AC)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["OpenLoopGain"])

    def test_loading_50_OP(self):
        # arrange
        filename = FIXTURES_DIR / "op.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Operating Point")
        self.assertFalse(qraw.complex)
        self.assertFalse(qraw.stepped)
        self.assertEqual(len(qraw.variables), 13)
        self.assertEqual(qraw.num_points, 31)
        self.assertEqual(len(qraw.variables), 13)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "s")
        self.assertEqual(qraw.variables[0].type, VariableType.PARAMETER)
        self.assertEqual(qraw.variables[8].index, 8)
        self.assertEqual(qraw.variables[8].name, "I(D1)")
        self.assertEqual(qraw.variables[8].type, VariableType.CURRENT)
        self.assertEqual(qraw.variables[12].index, 12)
        self.assertEqual(qraw.variables[12].name, "P(D1)")
        self.assertEqual(qraw.variables[12].type, VariableType.POWER)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 1.0e-15)
        self.assertEqual(qraw.abscissa_max, 1.0e-06)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables — two «...» groups each become their own PlotSuggestion
        self.assertEqual(len(qraw.plot_suggestions), 2)
        self.assertIs(qraw.plot_suggestions[0].chart_type, DC)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["V(vf1)", "I(D1)"])
        self.assertIs(qraw.plot_suggestions[1].chart_type, DC)
        self.assertEqual([v.name for v in qraw.plot_suggestions[1].variables], ["Id(J1)", "V(vd)"])

    def test_loading_70_VerilogBus(self):
        # arrange
        filename = FIXTURES_DIR / "VerilogBus.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertFalse(qraw.stepped)
        self.assertEqual(len(qraw.variables), 9)
        self.assertEqual(qraw.num_points, 1504)
        self.assertEqual(qraw.variables[0].index, 0)
        self.assertEqual(qraw.variables[0].name, "Time")
        self.assertEqual(qraw.variables[0].type, VariableType.TIME)
        self.assertEqual(qraw.variables[2].index, 2)
        self.assertEqual(qraw.variables[2].name, "V(outd[3])")
        self.assertEqual(qraw.variables[2].type, VariableType.VOLTAGE)
        self.assertEqual(qraw.variables[8].index, 8)
        self.assertEqual(qraw.variables[8].name, "I(V1)")
        self.assertEqual(qraw.variables[8].type, VariableType.CURRENT)
        # absissa
        self.assertEqual(len(qraw.variables[0].values), qraw.abscissa_points)
        # assert varable data lengths
        for variable in qraw.variables[1:]:
            self.assertEqual(len(variable.values), qraw.num_points)
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_min, 0.0)
        self.assertEqual(qraw.abscissa_max, 2.0e-05)
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables — five «tran ...» groups each become their own PlotSuggestion
        self.assertEqual(len(qraw.plot_suggestions), 5)
        self.assertIs(qraw.plot_suggestions[0].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[0].variables], ["V(outd[3])"])
        self.assertIs(qraw.plot_suggestions[1].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[1].variables], ["V(outd[2])"])
        self.assertIs(qraw.plot_suggestions[2].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[2].variables], ["V(outd[1])"])
        self.assertIs(qraw.plot_suggestions[3].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[3].variables], ["V(outd[0])"])
        self.assertIs(qraw.plot_suggestions[4].chart_type, TRANSIENT)
        self.assertEqual([v.name for v in qraw.plot_suggestions[4].variables], ["V(outa)"])
