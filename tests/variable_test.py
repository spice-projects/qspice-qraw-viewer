import unittest

import numpy as np

from viewer.variable import Variable, VariableType

class TestVariable(unittest.TestCase):

    def test_index(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        index_real = var_real.index
        index_complex = var_complex.index
        # assert
        self.assertEqual(index_real, 0)
        self.assertEqual(index_complex, 1)

    def test_name(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        name_real = var_real.name
        name_complex = var_complex.name
        # assert
        self.assertEqual(name_real, "Vout")
        self.assertEqual(name_complex, "Iin")

    def test_type(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        type_real = var_real.type
        type_complex = var_complex.type
        # assert
        self.assertEqual(type_real, VariableType.VOLTAGE)
        self.assertEqual(type_complex, VariableType.CURRENT)

    def test_values_property(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        values_real_result = var_real.values
        values_complex_result = var_complex.values
        # assert
        np.testing.assert_array_equal(values_real_result, values_real)
        np.testing.assert_array_equal(values_complex_result, values_complex)

    def test_steps(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        steps_real = var_real.steps
        steps_complex = var_complex.steps
        # assert
        self.assertEqual(steps_real, 2)
        self.assertEqual(steps_complex, 2)

    def test_complex_property(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        is_complex_real = var_real.complex
        is_complex_complex = var_complex.complex
        # assert
        self.assertFalse(is_complex_real)
        self.assertTrue(is_complex_complex)

    def test_step_values(self):
        # arrange
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        # act
        step0_real = var_real.step_values(0)
        step1_real = var_real.step_values(1)
        step0_complex = var_complex.step_values(0)
        step1_complex = var_complex.step_values(1)
        # assert
        np.testing.assert_array_equal(step0_real, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(step1_real, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(step0_complex, np.array([1+2j, 3+4j]))
        np.testing.assert_array_equal(step1_complex, np.array([5+6j, 7+8j]))
        # act & assert
        with self.assertRaises(IndexError):
            var_real.step_values(-1)
        with self.assertRaises(IndexError):
            var_real.step_values(2)

    def test_magnitude_property(self):
        # arrange
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        # act
        mag = var_complex.magnitude
        # assert
        np.testing.assert_array_almost_equal(mag.values, np.abs(values_complex))
        self.assertEqual(mag.name, "abs(Iin)")
        self.assertEqual(mag.type, VariableType.CURRENT)
        self.assertEqual(mag.steps, 2)
        # act & assert
        with self.assertRaises(ValueError):
            _ = var_real.magnitude

    def test_phase_property(self):
        # arrange
        values_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        values_real = np.array([1.0, 2.0, 3.0, 4.0])
        var_complex = Variable(1, "Iin", VariableType.CURRENT, values_complex, steps=2)
        var_real = Variable(0, "Vout", VariableType.VOLTAGE, values_real, steps=2)
        # act
        phase = var_complex.phase
        # assert
        np.testing.assert_array_almost_equal(phase.values, np.angle(values_complex, True))
        self.assertEqual(phase.name, "angle(Iin)")
        self.assertEqual(phase.type, VariableType.PHASE)
        self.assertEqual(phase.steps, 2)
        # act & assert
        with self.assertRaises(ValueError):
            _ = var_real.phase
