from pathlib import Path
from unittest import TestCase

from viewer.qraw_file import AbscissaScale, QRawFile

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
        self.assertEqual(qraw.steps, 6)
        self.assertEqual(len(qraw.expressions), 12)
        self.assertEqual(qraw.expressions[0].name, "V2")
        self.assertEqual(qraw.expressions[0].unit, "V")
        self.assertEqual(qraw.expressions[6].name, "Id(J1)")
        self.assertEqual(qraw.expressions[6].unit, "A")
        self.assertEqual(qraw.expressions[11].name, "P(V1)")
        self.assertEqual(qraw.expressions[11].unit, "W")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables to plot
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertIs(qraw.get_plot_suggestions()[0].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["Id(J1)"])

    def test_loading_20_AC(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 2)
        # 32 raw variables + 2 aliases: Freq and Omega
        self.assertEqual(len(qraw.expressions), 34)
        self.assertEqual(qraw.expressions[0].name, "Frequency")
        self.assertEqual(qraw.expressions[0].unit, "Hz")
        self.assertEqual(qraw.expressions[31].name, "X")
        self.assertEqual(qraw.expressions[32].name, "Freq")
        self.assertEqual(qraw.expressions[33].name, "Omega")
        self.assertEqual(qraw.expressions[31].unit, "")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables to plot (AC keyword maps to AC chart type)
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(vout)"])

    def test_loading_22_NyquistDia(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_Nyquist.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 24 raw variables + 5 alias resistor currents + 2 aliases Freq/Omega
        self.assertEqual(len(qraw.expressions), 31)
        self.assertEqual(qraw.expressions[0].name, "Frequency")
        self.assertEqual(qraw.expressions[0].unit, "Hz")
        self.assertEqual(qraw.expressions[1].name, "V(vout)")
        self.assertEqual(qraw.expressions[1].unit, "V")
        self.assertEqual(qraw.expressions[23].name, "I(COUT)")
        self.assertEqual(qraw.expressions[23].unit, "A")
        self.assertEqual(qraw.expressions[29].name, "Freq")
        self.assertEqual(qraw.expressions[30].name, "Omega")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual(qraw.get_plot_suggestions()[0].expressions, [])

    def test_loading_30_TRAN(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 47 raw variables + 7 alias resistor/conductance currents
        self.assertEqual(len(qraw.expressions), 54)
        self.assertEqual(qraw.expressions[0].name, "Time")
        self.assertEqual(qraw.expressions[0].unit, "s")
        self.assertEqual(qraw.expressions[1].name, "V(out)")
        self.assertEqual(qraw.expressions[1].unit, "V")
        self.assertEqual(qraw.expressions[46].name, "I(C1)")
        self.assertEqual(qraw.expressions[46].unit, "A")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 3)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(out)", "V(ss)", "V(in)"])
        self.assertEqual(qraw.get_plot_suggestions()[1].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["V(ics)", "V(comp)"])
        self.assertEqual(qraw.get_plot_suggestions()[2].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[2].expressions], ["V(on)", "V(off)"])

    def test_loading_40_Bode(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_Bode.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 2 raw variables + 2 aliases: Freq and Omega
        self.assertEqual(len(qraw.expressions), 4)
        self.assertEqual(qraw.expressions[0].name, "Frequency")
        self.assertEqual(qraw.expressions[0].unit, "Hz")
        self.assertEqual(qraw.expressions[1].name, "OpenLoopGain")
        self.assertEqual(qraw.expressions[2].name, "Freq")
        self.assertEqual(qraw.expressions[3].name, "Omega")
        self.assertEqual(qraw.expressions[1].unit, "V")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables to plot
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["OpenLoopGain"])

    def test_loading_50_OP(self):
        # arrange
        filename = FIXTURES_DIR / "op.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Operating Point")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        self.assertEqual(len(qraw.expressions), 13)
        self.assertEqual(len(qraw.expressions), 13)
        self.assertEqual(qraw.expressions[0].name, "s")
        self.assertEqual(qraw.expressions[0].unit, "")
        self.assertEqual(qraw.expressions[8].name, "I(D1)")
        self.assertEqual(qraw.expressions[8].unit, "A")
        self.assertEqual(qraw.expressions[12].name, "P(D1)")
        self.assertEqual(qraw.expressions[12].unit, "W")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 2)
        self.assertIs(qraw.get_plot_suggestions()[0].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(vf1)", "I(D1)"])
        self.assertIs(qraw.get_plot_suggestions()[1].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["Id(J1)", "V(vd)"])

    def test_loading_70_VerilogBus(self):
        # arrange
        filename = FIXTURES_DIR / "VerilogBus.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        self.assertEqual(len(qraw.expressions), 9)
        self.assertEqual(qraw.expressions[0].name, "Time")
        self.assertEqual(qraw.expressions[0].unit, "s")
        self.assertEqual(qraw.expressions[2].name, "V(outd[3])")
        self.assertEqual(qraw.expressions[2].unit, "V")
        self.assertEqual(qraw.expressions[8].name, "I(V1)")
        self.assertEqual(qraw.expressions[8].unit, "A")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 5)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(outd[3])"])
        self.assertEqual(qraw.get_plot_suggestions()[1].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["V(outd[2])"])
        self.assertEqual(qraw.get_plot_suggestions()[2].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[2].expressions], ["V(outd[1])"])
        self.assertEqual(qraw.get_plot_suggestions()[3].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[3].expressions], ["V(outd[0])"])
        self.assertEqual(qraw.get_plot_suggestions()[4].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[4].expressions], ["V(outa)"])
