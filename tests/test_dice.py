from scoring import DiceCoeff
import unittest
import numpy as np

class TestDiceCoeff(unittest.TestCase):
    def test_dicecoeff_init(self):
        dice = DiceCoeff()
        self.assertEqual(dice.maximum, 1)
        self.assertEqual(dice.minimum, 0)
        self.assertEqual(dice.is_lower_the_better, False)
        return

    def test_dicecoeff_init_name(self):
        dice = DiceCoeff(name='test0')
        self.assertEqual(dice.name, 'test0')
        return

    def test_dicecoeff_init_precision(self):
        dice = DiceCoeff(precision=19)
        self.assertEqual(dice.precision, 19)
        return

    def test_dicecoeff_check_dim(self):
        dice = DiceCoeff()
        array_0 = np.zeros((10, 9, 8))
        array_1 = np.zeros((10, 9, 8))
        array_2 = np.zeros((10, 9, 8, 1))
        array_3 = np.zeros((1))
        self.assertTrue(dice.check_y_pred_dimensions(array_0, array_1))
        self.assertFalse(dice.check_y_pred_dimensions(array_0, array_2))
        self.assertFalse(dice.check_y_pred_dimensions(array_0, array_3))
        return

    def test_dicecoeff_score(self):
        dice = DiceCoeff()
        array_fizz = np.zeros((10**3))
        array_buzz = np.zeros((10**3))
        # The slicing sets every 3rd entry to 1 for array_fizz, and every 5th for array_buzz
        # Knowing the size of the arrays, we know what the coefficient should be
        array_fizz[slice(0, None, 3)] = 1
        array_buzz[slice(0, None, 5)] = 1

        true_positives_fizzbuzz = np.divmod(len(array_fizz), 15)[0]+1

        div, rem = np.divmod(len(array_fizz), 3)
        array_fizz_pos = div + (rem > 0)

        div, rem = np.divmod(len(array_buzz), 5)
        array_buzz_pos = div + (rem > 0)

        expected_coef = 2*true_positives_fizzbuzz / (array_fizz_pos + array_buzz_pos)
        array_fizz_image = np.reshape(array_fizz, (10,10,10))
        array_buzz_image = np.reshape(array_buzz, (10,10,10))
        self.assertEqual(dice.calc_score(array_fizz_image, array_buzz_image), expected_coef)
        self.assertEqual(dice.calc_score(array_buzz_image, array_fizz_image), expected_coef)

        # While we're here; test the call
        # self.assertEqual(dice(array_fizz_image, array_buzz_image), expected_coef)
        return